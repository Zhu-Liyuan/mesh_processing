import os, argparse
from pathlib import Path
import point_cloud_utils as pcu
from tqdm import tqdm


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='Make mesh watertight')

parser.add_argument('-c', '--class_id')
args = parser.parse_args()


dataset_path = Path('/scratch/liyzhu/MA_Thesis/ShapeNetCore.v2/room_5cate_v2')

classes = os.listdir(dataset_path)
python_exc = '/scratch/liyzhu/miniconda3/envs/mesh_processing/bin/python'
scale_py = 'mesh-fusion/1_scale.py'
fusion_py = 'mesh-fusion/2_fusion.py'
simplify_py = 'mesh-fision/3_simplify.py'

# class_id = classes[int(args.class_id)]

for cls in classes:
    cls_path = dataset_path/cls
    instances = os.listdir(cls_path)
    for obj in tqdm(instances):
        obj_path = cls_path/obj/'model_normalized.obj'
        v, f = pcu.load_mesh_vf(obj_path)
        scale_path = cls_path/obj/'models'/'1_scale'
        depth_path = cls_path/obj/'models'/'2_depth'
        watertight = cls_path/obj/'models'/'3_watertight'
        # os.system(f'rm -rf {watertight}')
        os.system(f'{python_exc} {scale_py} --in_file {obj_path} --out_dir {scale_path}')
        os.system(f'{python_exc} {fusion_py} --mode=render --in_dir {scale_path} --out_dir {depth_path}')
        os.system(f'{python_exc} {fusion_py} --mode=fuse --in_dir {depth_path} --out_dir {watertight}')
        os.system(f'cp {watertight}/model_normalized.obj {cls_path/obj}/watertight.obj' )
        # os.system(f'rm -rf {scale_path}')
        # os.system(f'rm -rf {depth_path}')
        # os.system(f'rm -rf {watertight}')
        os.system(f'rm -rf {cls_path/obj}/models')
#     pass


        


    