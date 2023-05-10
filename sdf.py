import trimesh
import point_cloud_utils as pcu
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os

visualize = False
n_nss_samples = 50000
n_uni_samples = 50000

def gen_sdf_samples(mesh_path, visualize=False):
    ''' generating sdf samples for a watertight mesh'''
    v, f = pcu.load_mesh_vf(mesh_path)
    # n = pcu.estimate_mesh_vertex_normals(v, f)
    # Generate 10000 samples on a mesh with poisson disk samples
    # f_i are the face indices of each sample and bc are barycentric coordinates of the sample within a face
    fid_surf, bc_surf = pcu.sample_mesh_random(v, f, n_nss_samples)
    # pcu.sample_mesh_random()
    # Use the face indices and barycentric coordinate to compute sample positions and normals
    # v_poisson = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
    p_surf = pcu.interpolate_barycentric_coords(f, fid_surf, bc_surf, v)

    # n_poisson = pcu.interpolate_barycentric_coords(f, f_i, bc, n)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    # ns_sdfs = sample_sdf_near_surface(mesh, n_nss_samples)
    # colors = np.zeros_like(p_surf)
    # nss = trimesh.PointCloud(points,colors)

    noise = (np.random.standard_normal(size=p_surf.shape)) / (100)
    nss_pc = p_surf + noise

    nss_sdf, _, _ = pcu.signed_distance_to_mesh(nss_pc, v, f)

    # sample uniform samples in the pcu
    uni_pc = (np.random.rand(n_uni_samples, 3) - 0.5)
    uni_sdf, _, _  = pcu.signed_distance_to_mesh(uni_pc, v, f)

    
    if visualize == True:
        plt.hist(nss_sdf, bins=50)
        # plt.colorbar()
        plt.title('Histogram of Near-surface SDF Samples')
        plt.xlabel('SDF Values (near surface)')
        plt.ylabel('Counts')
        plt.show()
        
        cmap = matplotlib.cm.get_cmap('plasma')
        sdfs = np.asarray(nss_sdf)
        sdfs -= min(sdfs)
        sdfs /= np.max(sdfs)
        colors = [cmap(sd) for sd in sdfs]

        ns_pc = trimesh.PointCloud(nss_pc, colors)
        s = trimesh.Scene()
        s.add_geometry([ns_pc, mesh])
        s.show()
    return nss_pc, nss_sdf, uni_pc, uni_sdf


dataset_path = Path('/scratch/liyzhu/MA_Thesis/ShapeNetCore.v2/room_5cate')
loc = np.asarray([.0,.0,.0])
scale = 1
for cls in os.listdir(dataset_path):
    cls_path = dataset_path/cls
    for obj in os.listdir(cls_path):
        obj_path = cls_path/obj
        nss_pc, nss_sdf, uni_pc, uni_sdf = gen_sdf_samples(obj_path/'watertight.obj', visualize=True)
        nss_samples = np.concatenate([nss_pc, nss_sdf[:, None]], axis=1, dtype=np.float64)
        uni_samples = np.concatenate([uni_pc, uni_sdf[:, None]], axis=1)
        
        np.savez(obj_path/'points_nss.npz', loc = loc, points = nss_samples, scale=scale)
        np.savez(obj_path/'points_uni.npz', loc = loc, points = uni_samples, scale=scale)
        pass
    pass



