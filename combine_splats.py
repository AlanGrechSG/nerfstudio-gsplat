import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import plyfile
import torch
from tqdm import tqdm
import open3d as o3d
import argparse

def load_camera_intrinsics(transform_file: Path, device: torch.device = torch.device("cpu")):
    transform_json = json.loads(transform_file.read_text())
    cam_intrinsics = []
    for frame in transform_json["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])
        cam_intrinsics.append({
            "R": torch.tensor(transform_matrix[:3, :3], device=device), 
            "t": torch.tensor(transform_matrix[:3, 3], device=device),
            "fx": frame["fl_x"],
            "fy": frame["fl_y"],
            "cx": frame["cx"],
            "cy": frame["cy"],
            "W": frame["w"],
            "H": frame["h"],
        })

    return cam_intrinsics

def sphere_in_frustum(means_world, radii, R, t, fx, fy, cx, cy, W, H):
    # (N, 3) world → camera transform
    p_cam = (means_world - t) @ R.T
    X = p_cam[:, 0]
    Y = p_cam[:, 1]
    Z = p_cam[:, 2]

    r = radii

    # 1. behind camera test: Z + r > 0
    mask = (Z + r) > 0

    # Early exit if none pass
    if not mask.any():
        return mask

    # Filter values
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]
    r = r[mask]

    # 2. Compute sphere-extent projections
    # Horizontal
    u_min = fx * (X - r) / Z + cx
    u_max = fx * (X + r) / Z + cx

    # Vertical
    v_min = fy * (Y - r) / Z + cy
    v_max = fy * (Y + r) / Z + cy

    # 3. Frustum intersection conditions
    horiz_ok = (u_max >= 0) & (u_min <= W)
    vert_ok  = (v_max >= 0) & (v_min <= H)

    inside = horiz_ok & vert_ok

    # Return mask in original indexing
    full = torch.zeros_like(mask)
    full[mask] = inside

    return full


def main():
    if platform.system() == "Linux":
        data_dir = Path("/mnt/d/full-cathedral")
    else:
        data_dir = Path("D:\\full-cathedral")

    processed_folder = "processed"
    splat_files = list((data_dir / "exports").glob("splat_*.ply"))

    def load_ply_verts(path):
        ply = plyfile.PlyData.read(path)
        return ply["vertex"]
        
    print("Loading splat files...")
    all_verts = [load_ply_verts(splat_file) for splat_file in splat_files]

    # Remove points that are not in camera frustums
    transform_files = [data_dir / processed_folder / f"transforms_{f.stem.split('_')[-1]}.json" for f in splat_files]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, transform_file in tqdm(enumerate(transform_files), desc="Filtering splats by camera frustum", total=len(transform_files)):
        splat = all_verts[i]
        means_world = torch.tensor(np.vstack((splat["x"], splat["y"], splat["z"])).T, device=device)
        radii = torch.tensor(np.abs(np.vstack([splat["scale_0"], splat["scale_1"], splat["scale_2"]])).max(axis=0), device=device)

        cam_intrinsics = load_camera_intrinsics(transform_file, device=device)
        visibility = torch.zeros(len(splat), dtype=torch.bool, device="cpu")
        for cam in cam_intrinsics:
            visibility |= sphere_in_frustum(means_world, radii, 
                                            cam["R"], cam["t"], 
                                            fx=cam["fx"], fy=cam["fy"], 
                                            cx=cam["cx"], cy=cam["cy"], 
                                            W=cam["W"], H=cam["H"]).cpu()
            
        num_removed = (~visibility).sum().item()
        print("Removed", num_removed, "out of", len(splat), "splats for cluster", i)
        
        visibility_mask = visibility.numpy()
        all_verts[i] = splat[visibility_mask]

    base = all_verts[0]
    dtype = base.dtype

    total = sum(len(verts) for verts in all_verts)
    merged = np.empty(total, dtype=dtype)

    idx = 0
    for v in tqdm(all_verts, desc="Merging splats", total=len(all_verts)):
        merged[idx:idx+len(v)] = v
        idx += len(v)

    # merged = np.concatenate(all_verts)

    print("Saving combined splat PLY...")
    vertex_el = plyfile.PlyElement.describe(merged, "vertex")
    plyfile.PlyData([vertex_el]).write(data_dir / "exports" / "splat_combined.ply")

    # retrain_all(data_dir, processed_folder)

def retrain_all(data_dir: Path, splat_ply: Path, processed_folder: str = "processed"):
    retrain_all_cmd = [
        "ns-train", "splatfacto",
        "--data", processed_folder,
        "--output-dir", "outputs/splat_combined",
        "--viewer.quit-on-train-completion", "True",
        "--mixed-precision", "True",
        "--pipeline.datamanager.cache-images", "cpu",
        "--pipeline.model.gaussian-init-ply", str(data_dir / splat_ply),
        "nerfstudio-data",
        "--downscale-factor", "4"
    ]

    subprocess.run(retrain_all_cmd, cwd=data_dir)

    gs_config_path = list((data_dir / f"outputs/splat_combined/{processed_folder}/splatfacto").rglob("**/config.yml"))[-1]

    export_command = [
        "ns-export", "gaussian-splat",
        "--load-config", str(gs_config_path),
        "--output-dir", "exports",
        "--output-filename", "splat_combined_retrained.ply"
    ]

    subprocess.run(export_command, cwd=data_dir)

def view_splatcloud():
    if platform.system() == "Linux":
        data_dir = Path("/mnt/d/full-cathedral")
    else:
        data_dir = Path("D:\\full-cathedral")

    splat_file = data_dir / "exports" / "splat_combined.ply"

    pcd = o3d.io.read_point_cloud(str(splat_file))
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine splat point clouds and retrain.")
    parser.add_argument("--data", required=True, help="The path to the base data directory.")
    parser.add_argument("--splat_ply", required=True, help="The path to the combined splat PLY file relative to the base data directory.")
    args = parser.parse_args()
    retrain_all(Path(args.data), Path(args.splat_ply))