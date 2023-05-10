# Mesh processing workflow
- Use mesh-fusion to make mesh watertight. (Manifold-2 or point cloud utils can also work but mesh-fusion performs the best)
- Use GAPS or DeepSDF processing pipeline to generate near surface SDF samples
- Use point cloud utils to generate uniform SDF samples
- Use pyrender to render depth maps for watertight meshes
- Backproject depth map into point cloud and transform into world frame



``` bash
python watertight.py
```

```
python sample_sdf.py
```

```
python render_depth.py
```
