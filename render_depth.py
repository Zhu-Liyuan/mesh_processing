# render depth for a single mesh 
# code adapted from mesh fusion
# Liyuan Zhu
# 2023.May.7
import os
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path
from tqdm import tqdm

n_views = 24
image_height = 100
image_width = 100
fx = 100
fy = 100
cx = 50
cy = 50
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32)
k = np.reshape(k, (3, 3))
visualize = False

def gen_random_poses(n_views):
    cam_poses = []
    
    for _ in range(n_views):
        Rot_mat = Rotation.random().as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = Rot_mat
        xyz = Rot_mat @ np.linalg.inv(k) @ np.array([[cx],[cy],[1]])
        pose[:3, [3]] = xyz * 1.1
        cam_poses.append(pose)
    return cam_poses


def render_depth(mesh_path, n_views):
    
    cam_poses = gen_random_poses(n_views)
    
    # camera = pyrender.PerspectiveCamera(yfov=2 * np.arctan(fy/(image_height/2)), aspectRatio=image_height/image_width)
    scene = pyrender.Scene()
    tri_mesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    
    depth_maps = []
    pcls = []
    T_cw_list = []
    
    
    if visualize:
        tri_mesh.show()
    
    for pose in cam_poses:
        scene.clear()
        # camera = pyrender.PerspectiveCamera(yfov=2 * np.arctan(1/2.), aspectRatio=1.0)
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        
        scene.add(mesh)
        scene.add(camera, pose=pose)
        r = pyrender.OffscreenRenderer(image_height, image_width)

        rgb, depth = r.render(scene)
        r.delete()
        depth_maps.append(depth)
        
        if visualize:
            plt.imshow(depth, cmap="plasma")
            plt.axis('off')
            plt.colorbar()
            plt.show()
        
        pcl_cam = pointcloud(depth)
        
        # https://github.com/colmap/colmap/issues/704#issuecomment-954161261 pyrender pose different from colmap
        R = pose[:3, :3]
        T = pose[:3, [3]]
        T[:,1:3] *= -1
        pcl_cam[:,1:3] *= -1
        pcl = R @ (pcl_cam.T ) + T
        pcl = pcl.T
        pcls.append(pcl)
        
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, [3]] = T
        T_cw_list.append(T_cw)
        # pcl = pose[:3,:3].T @ (pcl_cam.T ) - pose[:3,[3]]

        # debug
        # s = trimesh.Scene()
        # s.add_geometry([trimesh.PointCloud(pcl), tri_mesh])
        # s.show()
        
        
    return pcls, T_cw_list

def pointcloud(depth):
        # fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)
        False
        d = depth[mask]
        x = mask[1] * d
        y = mask[0] * d
        
        uvd = np.concatenate([x[:, None], y[:, None], d[:, None]], axis=1)
        # xyz_c = (np.linalg.inv(k) @ uv.T).T
        # xyz_c = np.concatenate([xyz_c, np.ones_like(d[:, None])], axis=1)
        xyz_c = (np.linalg.inv(k) @ uvd.T).T
        # xyz_c[:,1] = 1 - xyz_c[:,1]
        # xyz_c *= d[:, None]
        
        return xyz_c


if __name__ == "__main__":
    # cam_poses = gen_random_poses(n_views)
    pcl_list, T_cw_list = render_depth('watertight.obj', 24)

    if visualize:
        pcls = np.concatenate(pcl_list[:3],axis=0)
        pcls = trimesh.PointCloud(pcls)
        s = trimesh.Scene()
        s.add_geometry([pcls, trimesh.load('watertight.obj')])
        s.show()
    
    shapenet_path = Path('/scratch/liyzhu/MA_Thesis/ShapeNetCore.v2/room_5cate_v2')
    samples_path = Path('/scratch/liyzhu/MA_Thesis/ShapeNetCore.v2/room_5cate_v2_sdf')
    
    # classes = os.listdir(shapenet_path)
    for cls in os.listdir(shapenet_path):
        cls_path = shapenet_path/cls
        cls_dep_path = samples_path/(str(cls)+'_dep')
        # os.makedirs(cls_dep_path, exist_ok=True)
        for obj in tqdm(os.listdir(cls_path)):
            obj_path = cls_path/obj/'watertight.obj'
            pcl_list, T_cw_list = render_depth(obj_path, 24)
            out_path = cls_dep_path/obj
            os.makedirs(out_path, exist_ok=True)
            for i, (p_w, T_cw) in enumerate(zip(pcl_list, T_cw_list)):
                np.savez(out_path/f'dep_pcl_{i}', p_w=p_w, T_cw=T_cw)
            
    
    
