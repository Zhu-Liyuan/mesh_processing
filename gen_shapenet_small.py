import os, random
from pathlib import Path
import trimesh
from tqdm import tqdm
import numpy as np
import point_cloud_utils as pcu

shapenet_small_path = Path('/scratch/liyzhu/MA_Thesis/ShapeNetCore.v2/room_5cate_v2')
shapenet_full = Path('/scratch/liyzhu/MA_Thesis/ShapeNetCore.v2/ShapeNetCore.v2')

classes = ['04256520', '03636649', '03001627', '04379243', '02933112']
classes.sort()
class_size = 250
split_percentages = [.7, .1, .2]
n_obj = [int(p * class_size) for p in split_percentages]
split_names = ['train', 'val', 'test']


for cls in classes:
    cls_path = shapenet_full/cls
    os.makedirs(shapenet_small_path/cls, exist_ok=True)
    objects = os.listdir(cls_path)
    cls_samples = random.sample(objects, class_size)
    
    for sample in cls_samples:
        os.makedirs(shapenet_small_path/cls/sample)
        sample_path = cls_path/sample/'models/model_normalized.obj'
        v,f = pcu.load_mesh_vf(str(sample_path))
        pcu.save_mesh_vf(str(shapenet_small_path/cls/sample/'model_normalized.obj'), v=v, f=f)
    
    # for obj_id in tqdm(os.listdir(cls_path)):
    #     mesh = trimesh.load(shapenet_full/cls/obj_id/'models/model_normalized.obj')
    #     mesh.export(shapenet_small_path/cls/obj_id/'model_normalized.obj')
    