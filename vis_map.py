# Source - https://stackoverflow.com/a
# Posted by Chee Loong Soon, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-14, License - CC BY-SA 4.0

from pathlib import Path
import numpy as np
import open3d as o3d
import torch
import json
import colorsys as colorss

# Read .ply file
# input_file = "C:/Dev/VGGT-Long/exps/_mnt_d_st-johns_test-data_chapel-06-germany/2025-11-11-16-32-04/pcd/combined_pcd.ply"
# pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

def farthest_point_sampling_colors(n):
    colors = np.random.rand(1, 3)
    while len(colors) < n:
        candidates = np.random.rand(1000, 3)
        min_dists = np.min(np.sqrt(((candidates[:, None] - colors)**2).sum(-1)), axis=1)
        best = candidates[np.argmax(min_dists)]
        colors = np.vstack([colors, best])
    return colors

def view_multi_pointclouds():
    pcd_dir = Path("D:\\full-cathedral\\processed")
    pcd_files = list(pcd_dir.glob("sparse_pc_*.ply"))

    merged = o3d.geometry.PointCloud()

    cols = farthest_point_sampling_colors(len(pcd_files))

    for pcd_file, color in zip(pcd_files, cols):
        pcd = o3d.io.read_point_cloud(str(pcd_file)) # Read the point cloud
        points = np.asarray(pcd.points)
        n = 10_000
        idx = np.random.choice(len(points), size=min(n, len(points)), replace=False)

        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(points[idx])
        sample_pcd.paint_uniform_color(color)

        # pcd.paint_uniform_color(color)  # Assign a random color to the point cloud

        merged += sample_pcd

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([merged])

def view_multi_gs():
    pcd_dir = Path("D:\\full-cathedral\\exports")
    pcd_files = list(pcd_dir.glob("splat_*.ply"))[0:2]

    merged = o3d.geometry.PointCloud()

    cols = farthest_point_sampling_colors(len(pcd_files))

    for pcd_file, color in zip(pcd_files, cols):
        pcd = o3d.io.read_point_cloud(str(pcd_file)) # Read the point cloud
        points = np.asarray(pcd.points)
        n = 100_000
        idx = np.random.choice(len(points), size=min(n, len(points)), replace=False)

        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(points[idx])
        sample_pcd.paint_uniform_color(color)

        # pcd.paint_uniform_color(color)  # Assign a random color to the point cloud

        merged += sample_pcd

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([merged])

def view_pointcloud():
    pcd_file = Path("D:\\full-cathedral\\processed\\sparse_pc.ply")
    pcd = o3d.io.read_point_cloud(str(pcd_file)) # Read the point cloud
    points = np.asarray(pcd.points)
    n = 1_000_000
    idx = np.random.choice(len(points), size=min(n, len(points)), replace=False)

    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(points[idx])

    o3d.visualization.draw_geometries([sample_pcd])

def view_multi_transforms():
    transforms_dir = Path("D:\\full-cathedral\\processed")
    transform_files = list(transforms_dir.glob("transforms_*.json"))
    all_points = []
    all_colors = []

    cols = farthest_point_sampling_colors(len(transform_files))

    for transform_file, color in zip(transform_files, cols):
        with open(transform_file, 'r') as f:
            transform_data = json.load(f)

        frames = transform_data["frames"]

        transforms = np.array([f["transform_matrix"] for f in frames])

        all_points.append(transforms[:, :3, 3])
        all_colors.append(np.tile(color, (len(transforms), 1)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([pcd])

def view_transforms():
    transform_file = "E:\\v001\\processed\\transforms.json"
    with open(transform_file, 'r') as f:
        transform_data = json.load(f)

    frames = transform_data["frames"]

    transforms = np.array([f["transform_matrix"] for f in frames])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transforms[:, :3, 3])

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([pcd]) 

if __name__ == "__main__":
    view_multi_gs()