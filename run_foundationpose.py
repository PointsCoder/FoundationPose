#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-contained demo for FoundationPose on MegaSAM outputs.

功能：
- 从 outputs/<key>/geometry/reconstruction/sgd_cvd_hr.npz 读取逐帧 RGB、Depth、K
- 从 outputs/<key>/scene.json 读取对象 mesh 与第0帧对象 mask 路径
- 首帧用 FoundationPose.register() 完成注册，后续帧用 track_one() 追踪
- 保存每帧的位姿到 debug/ob_in_cam/<frame_id>.txt，并可保存渲染可视化到 debug/track_vis/

作者：基于官方 run_demo.py 的接口，整合 MegaSAM 数据读取
"""

import os, json, glob, argparse, logging
from pathlib import Path
import numpy as np
import cv2, imageio
import trimesh

# 依赖 FoundationPose 仓库：estimater.py 内暴露的类与工具函数
from estimater import *             # FoundationPose, ScorePredictor, PoseRefinePredictor, draw_posed_3d_box, draw_xyz_axis, depth2xyzmap, toOpen3dCloud
import nvdiffrast.torch as dr       # 光栅化上下文

# ------------------------- 日志 -------------------------
def set_logging():
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO
    )

# ------------------------- 工具 -------------------------
def load_megasam_npz(npz_path: Path):
    """
    读取 MegaSAM 的 npz（示例键：images[N,H,W,3], depths[N,H,W], intrinsic 或 intrinsics）。
    兼容：
      - K 为 (3,3)（全局一致）或 (N,3,3)（逐帧）。
    返回：
      imgs:   np.uint8 [N,H,W,3]
      depths: float32  [N,H,W]  单位：米（若不是米，下游按米使用）
      Ks:     float32  [N,3,3]  逐帧内参（若输入只有一个K，则广播）
    """
    data = np.load(npz_path, allow_pickle=True)
    # 取 image / depth
    if "images" in data:
        imgs = data["images"]
    elif "rgbs" in data:
        imgs = data["rgbs"]
    else:
        raise KeyError(f"{npz_path} 中未找到 'images' 或 'rgbs'")
    depths = data["depths"] if "depths" in data else data["depth"]
    # 取 K
    if "intrinsic" in data:
        K_raw = data["intrinsic"].astype(np.float32)
    elif "intrinsics" in data:
        K_raw = data["intrinsics"].astype(np.float32)
    else:
        raise KeyError(f"{npz_path} 中未找到 'intrinsic(s)'")

    imgs = imgs.astype(np.uint8)
    depths = depths.astype(np.float32)

    N = len(imgs)
    if K_raw.shape == (3, 3):
        Ks = np.repeat(K_raw[None, ...], N, axis=0)
    elif K_raw.shape[0] == N:
        Ks = K_raw
    else:
        raise ValueError(f"Intrinsic 形状不匹配：{K_raw.shape} vs N={N}")

    return imgs, depths, Ks


def choose_object_and_paths(scene_json: Path, prefer_labels=None, prefer_oid=None):
    """
    从 scene.json 中挑选一个对象，并返回 mesh 路径与 mask 路径。
    优先顺序（mesh）：'refined' > 'aligned' > 'glb'
    优先对象：按 prefer_oid / prefer_labels 指定，否则拿第一个
    返回：mesh_path, mask_path, meta_obj (dict)
    """
    meta = json.loads(Path(scene_json).read_text())
    objs = meta.get("objects", [])
    if len(objs) == 0:
        raise FileNotFoundError(f"{scene_json} 中 objects 为空")

    def pick_obj_index():
        # 先按 oid
        if prefer_oid is not None:
            for i, o in enumerate(objs):
                if int(o.get("oid", -1)) == int(prefer_oid):
                    return i
        # 再按 label
        if prefer_labels:
            labels = [prefer_labels] if isinstance(prefer_labels, str) else list(prefer_labels)
            labels = [s.lower() for s in labels]
            for i, o in enumerate(objs):
                if str(o.get("label", "")).lower() in labels:
                    return i
        # 默认第一个
        return 0

    idx = pick_obj_index()
    o = objs[idx]


    mesh_cands = [o.get("aligned")]
    mesh_path = next((p for p in mesh_cands if p and Path(p).exists()), None)
    if mesh_path is None:
        raise FileNotFoundError(f"未在对象条目中找到可用 mesh：{mesh_cands}")

    mask_path = o.get("mask", None)
    if mask_path is None or not Path(mask_path).exists():
        # 退化到场景级 mask.jpg
        scene_dir = Path(scene_json).parent
        fallback = scene_dir / "mask.jpg"
        if not fallback.exists():
            raise FileNotFoundError(f"对象条目无 'mask' 且 fallback {fallback} 不存在")
        mask_path = str(fallback)

    return mesh_path, mask_path, o, meta, idx


def resize_rgb(rgb, W, H):
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

def resize_depth(depth, W, H):
    return cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

def resize_mask(msk, W, H):
    return cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

# ------------------------- 内嵌数据读取器 -------------------------
class MegaSAMReaderInMemory:
    """
    适配 FoundationPose 的简单 DataReader：
      - get_color(i)  -> RGB[H,W,3] uint8
      - get_depth(i)  -> depth[H,W] float32 (米)
      - get_mask(i)   -> 第0帧返回对象mask，其余帧返回全零；FP 只需要首帧 mask
      - K             -> 3x3 内参（若逐帧不同，这里也保留每帧一份）
      - id_strs       -> ['000000', '000001', ...]
    支持 shorter_side 下采样，并同步缩放 K。
    """
    def __init__(self, imgs, depths, Ks, obj_mask_path, shorter_side=None, zfar=np.inf):
        assert imgs.ndim == 4 and imgs.shape[-1] == 3, "imgs 必须为 [N,H,W,3]"
        assert depths.ndim == 3, "depths 必须为 [N,H,W]"
        assert Ks.ndim == 3 and Ks.shape[1:] == (3,3), "Ks 必须为 [N,3,3]"

        self.N, self.H0, self.W0 = imgs.shape[0], imgs.shape[1], imgs.shape[2]

        # 计算缩放
        self.downscale = 1.0
        if shorter_side is not None:
            self.downscale = float(shorter_side) / float(min(self.H0, self.W0))
        self.H = int(round(self.H0 * self.downscale))
        self.W = int(round(self.W0 * self.downscale))

        # 缩放图像、深度、K
        self.colors = np.stack([resize_rgb(imgs[i], self.W, self.H) for i in range(self.N)], axis=0)
        self.depths = np.stack([resize_depth(depths[i], self.W, self.H) for i in range(self.N)], axis=0)

        self.Ks = Ks.copy().astype(np.float32)
        self.Ks[:, :2, :] *= self.downscale  # fx,fy,cx,cy 缩放

        # 深度清理
        for i in range(self.N):
            d = self.depths[i]
            d[(d < 1e-3) | (d >= zfar)] = 0.0
            self.depths[i] = d

        # mask（只用于第0帧）
        m0 = cv2.imread(str(obj_mask_path), cv2.IMREAD_GRAYSCALE)
        if m0 is None:
            raise FileNotFoundError(f"mask 文件不存在：{obj_mask_path}")
        m0 = (m0 > 127).astype(np.uint8)
        m0 = resize_mask(m0, self.W, self.H)
        self.masks = [m0] + [np.zeros((self.H, self.W), dtype=np.uint8) for _ in range(self.N - 1)]

        # 帧 ID 字符串
        self.id_strs = [f"{i:06d}" for i in range(self.N)]

    def __len__(self):
        return self.N

    @property
    def K(self):
        # 若逐帧不同，这里返回首帧；FoundationPose 接口是传入 K=...，后续追踪同样传 K。
        # 为稳妥起见，外层调用时对每帧使用对应 self.Ks[i]
        return self.Ks[0]

    def get_color(self, i):
        return self.colors[i]

    def get_depth(self, i):
        return self.depths[i]

    def get_mask(self, i):
        return self.masks[i]

def convert_glb_to_obj_temp(glb_path: str):
    """
    若输入是 .glb/.gltf：
      - 读取 GLB，强制提取纹理（material.image 或 baseColorTexture）；
      - 导出到临时目录：textured_simple.obj + textured_simple.obj.mtl + texture_map.png；
      - 返回 (obj_path, tmpdir)。
    若不是 .glb/.gltf：返回 (glb_path, None)。
    约束：如果未检测到纹理，抛出异常。
    依赖：trimesh, Pillow
    """
    from pathlib import Path
    import tempfile, shutil
    import numpy as np
    from PIL import Image
    from trimesh.visual.material import SimpleMaterial

    p = Path(glb_path)
    if p.suffix.lower() not in ['.glb', '.gltf']:
        return glb_path, None

    tmpdir = tempfile.mkdtemp(prefix='fp_glb2obj_')      # 临时目录
    obj_path = Path(tmpdir) / 'textured_simple.obj'      # 与官方示例一致的命名
    tex_path = Path(tmpdir) / 'texture_map.png'

    # 读取 GLB
    m = trimesh.load(str(p), force='mesh')

    # 提取纹理（必须存在）
    mat = getattr(m.visual, 'material', None)
    tex_img = None
    if mat is not None:
        # 情况1：SimpleMaterial.image
        if hasattr(mat, 'image') and mat.image is not None:
            if isinstance(mat.image, Image.Image):
                tex_img = mat.image
            else:
                arr = np.asarray(mat.image)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                tex_img = Image.fromarray(arr)
        # 情况2：PBRMaterial.baseColorTexture
        elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            bct = mat.baseColorTexture
            if isinstance(bct, Image.Image):
                tex_img = bct
            else:
                arr = np.asarray(bct)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                tex_img = Image.fromarray(arr)

    if tex_img is None:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"[convert_glb_to_obj_temp] GLB 缺少纹理（material.image 或 baseColorTexture）：{glb_path}")

    # 保存纹理为固定文件名，确保 .mtl 中 map_Kd 能引用
    tex_img.save(str(tex_path))

    # 绑定 SimpleMaterial（确保有 material.image 且 .mtl 写入 map_Kd texture_map.png）
    m.visual.material = SimpleMaterial(image=str(tex_path))

    # 导出 OBJ（trimesh 会写 .mtl，并引用 texture_map.png）
    m.export(str(obj_path))
    logging.info(f"[convert_glb_to_obj_temp] 导出临时 OBJ: {obj_path}")
    return str(obj_path), tmpdir


# ------------------------- 主流程 -------------------------
def main():
    set_logging()
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/jmao/project/pi-I/")
    parser.add_argument("--key", type=str, default="video_pen")

    # FoundationPose 参数
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--debug_dir", type=str, default="debug")

    # 下采样与远裁剪
    parser.add_argument("--shorter_side", type=int, default=None, help="如果设置，则按最短边缩放到该值")
    parser.add_argument("--zfar", type=float, default=float("inf"))

    # 可选：指定对象 label 或 oid
    parser.add_argument("--object_label", type=str, default=None)
    parser.add_argument("--object_oid", type=int, default=None)

    # 可选：直接指定 mesh/mask/npz 路径（否则按默认规则从 key 目录推断）
    parser.add_argument("--mesh_file", type=str, default=None)
    parser.add_argument("--mask_file", type=str, default=None)
    parser.add_argument("--npz_file",  type=str, default=None)
    parser.add_argument("--scene_json", type=str, default=None)

    args = parser.parse_args()

    # 目录组织（与你的示例保持一致）
    base = Path(args.base_dir)
    key  = args.key
    out_scene_dir = base / f"outputs/{key}/scene"
    npz_p  = Path(args.npz_file) if args.npz_file else base / f"outputs/{key}/geometry/reconstruction/sgd_cvd_hr.npz"
    scene_json = Path(args.scene_json) if args.scene_json else out_scene_dir / "scene.json"

    debug_dir = Path(args.debug_dir)
    (debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)
    (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)

    logging.info(f"读取 MegaSAM npz: {npz_p}")
    imgs, depths, Ks = load_megasam_npz(npz_p)
    N, H0, W0 = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    logging.info(f"帧数 N={N}, 原始分辨率 {W0}x{H0}")

    # 选择对象与 mask/mesh
    if args.mesh_file and args.mask_file:
        mesh_path = args.mesh_file
        mask_path = args.mask_file
        meta_obj, meta = None, None
        logging.info(f"使用用户提供的 mesh/mask: {mesh_path}, {mask_path}")
    else:
        logging.info(f"读取 scene.json: {scene_json}")
        mesh_path, mask_path, meta_obj, scene_dict, obj_id = choose_object_and_paths(
            scene_json,
            prefer_labels=args.object_label,
            prefer_oid=args.object_oid
        )
        logging.info(f"选择对象：label={meta_obj.get('label','?')}, oid={meta_obj.get('oid','?')}")
        logging.info(f"mesh={mesh_path}")
        logging.info(f"mask={mask_path}")

    obj_tmpdir = None
    mesh_path, obj_tmpdir = convert_glb_to_obj_temp(mesh_path)

    # 构建 Reader
    reader = MegaSAMReaderInMemory(
        imgs=imgs, depths=depths, Ks=Ks,
        obj_mask_path=mask_path,
        shorter_side=args.shorter_side,
        zfar=args.zfar
    )
    H, W = reader.H, reader.W
    logging.info(f"实际使用分辨率：{W}x{H}  (downscale={reader.downscale:.3f})")

    # 加载 mesh
    mesh = trimesh.load(mesh_path)


    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # 初始化 FoundationPose
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx
    )
    logging.info("FoundationPose 初始化完成")

    # 逐帧处理：首帧 register，后续 track
    all_poses = []
    last_pose = None
    for i in range(len(reader)):
        K_i   = reader.Ks[i].astype(np.float64, copy=False)
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        logging.info(f"处理帧 i={i}/{len(reader)-1}")
        if i == 0:
            ob_mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=K_i, rgb=color, depth=depth, ob_mask=ob_mask, iteration=args.est_refine_iter)
            if True:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

        else:
            pose = est.track_one(rgb=color, depth=depth, K=K_i, iteration=args.track_refine_iter)

        last_pose = pose

        # 保存位姿
        np.savetxt(debug_dir / "ob_in_cam" / f"{reader.id_strs[i]}.txt", pose.reshape(4,4))
        all_poses.append(pose.reshape(4, 4))

        # 可视化
        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K_i, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_i, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)

            if args.debug >= 2:
                imageio.imwrite(debug_dir / "track_vis" / f"{reader.id_strs[i]}.png", vis)

    logging.info("全部帧处理完成。输出：")
    logging.info(f" - 位姿：{debug_dir/'ob_in_cam'}")
    if args.debug >= 2:
        logging.info(f" - 可视化：{debug_dir/'track_vis'}")

    all_poses = np.stack(all_poses, axis=0)
    obj_info = scene_dict["objects"][obj_id]
    pose_save_path = base / f"outputs/{key}/scene/assets" / f"{obj_info['oid']}_{obj_info['label']}_trajs.npy"
    np.save(pose_save_path, all_poses)
    scene_dict["objects"][obj_id]["trajs"] = str(pose_save_path)
    with open(scene_json, 'w') as f:
        json.dump(scene_dict, f, indent=2)
    logging.info(f"位姿已保存到 {pose_save_path}")

    if obj_tmpdir is not None and Path(obj_tmpdir).exists():
        import shutil
        shutil.rmtree(obj_tmpdir, ignore_errors=True)
        logging.info(f"[convert_glb_to_obj_temp] 已清理临时目录：{obj_tmpdir}")


if __name__ == "__main__":
    main()
