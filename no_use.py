from mesh_to_sdf import sample_sdf_near_surface
import point_cloud_utils as pcu
import trimesh
import pyrender
import numpy as np
import mesh_to_sdf


def compute_unit_sphere_transform(mesh: trimesh.Trimesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

# mesh = trimesh.load('input.obj')
mesh_path = 'input.obj'

v, f = pcu.load_mesh_vf("input.obj")
resolution = 50_000

# make the original mesh watertight
v, f = pcu.make_mesh_watertight(v, f, resolution)
mesh = trimesh.Trimesh(vertices=v, faces=f)

# n_points = 100000
# n = pcu.estimate_mesh_vertex_normals(v, f)

# sample points and tra
points, sdf = sample_sdf_near_surface(mesh, number_of_points=500000)
translation, scale = compute_unit_sphere_transform(mesh)
points = (points / scale) - translation
sdf /= scale

nss_mask = abs(sdf) <= 0.05
points = points[nss_mask]
sdf = sdf[nss_mask]

n_samples = 100000
samples_mask = np.random.choice(np.arange(len(sdf)), replace=False)
points


colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)