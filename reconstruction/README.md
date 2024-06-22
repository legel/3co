Processing pipeline from iPhone Pro to 3D mesh generation:
1. Camera calibration and LiDAR data is collected from an iPhone Pro using custom app.
2. That is processed with code here (foundry.py) to produce a custom low-entropy neural scene representation that is very close to a surface.
3. Points are sampled in 3D from every image into discrete 3D point clouds using pointcloud_processing.py.
4. This is converted into a 3D mesh representation using mesh_extraction.py.
5. The mesh then has UV textures that are differentiably optimized through a PyTorch3D inverse rendering process in inverse_rendering.py.
