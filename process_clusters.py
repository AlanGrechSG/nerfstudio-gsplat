import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

def compute_pointclouds():
    if platform.system() == "Linux":
        data_dir = Path("/mnt/d/full-cathedral")
    else:
        data_dir = Path("D:\\full-cathedral")

    processed_dir = data_dir / "processed"
    cluster_transforms = list(processed_dir.glob("transforms_*.json"))

    if (processed_dir / "transforms.json").exists():
        os.rename(processed_dir / "transforms.json", processed_dir / "transforms_all.json")

    for transform_path in cluster_transforms:
        cluster_id = transform_path.stem.split("_")[-1]
        os.rename(transform_path, processed_dir / "transforms.json")

        nerf_train_command = [
            "ns-train", "nerfacto",
            "--data", "processed",
            "--output-dir", f"outputs/cluster_{cluster_id}",
            "--viewer.quit-on-train-completion", "True",
            "--max-num-iterations", "20000",
            "nerfstudio-data",
            "--downscale-factor", "4"
        ]

        subprocess.run(nerf_train_command, cwd=data_dir)

        # --output-dir exports/pcd/ --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False 

        nerf_config_path = list((data_dir / f"outputs/cluster_{cluster_id}/processed/nerfacto").rglob("**/config.yml"))[0]

        export_pcd_command = [
            "ns-export", "pointcloud",
            "--load-config", str(nerf_config_path),
            "--output-dir", "exports/pcd/",
            "--num-points", "1000000",
            "--remove-outliers", "True",
            "--normal-method", "open3d",
            "--save-world-frame", "True"
        ]

        subprocess.run(export_pcd_command, cwd=data_dir)
        os.rename(data_dir / f"exports/pcd/point_cloud.ply", data_dir / f"exports/pcd/pcd_{cluster_id}.ply")
        shutil.move(data_dir / f"exports/pcd/pcd_{cluster_id}.ply", processed_dir / f"pcd_{cluster_id}.ply")

        transform_json = json.loads((processed_dir / "transforms.json").read_text())

        transform_json["ply_file_path"] = f"pcd_{cluster_id}.ply"

        with open(processed_dir / "transforms.json", "w") as f:
            json.dump(transform_json, f, indent=4)

        os.rename(processed_dir / "transforms.json", transform_path)

    os.rename(processed_dir / "transforms_all.json", processed_dir / "transforms.json")


def main():
    if platform.system() == "Linux":
        data_dir = Path("/mnt/d/full-cathedral")
    else:
        data_dir = Path("D:\\full-cathedral")

    processed_folder = "processed"
    processed_dir = data_dir / processed_folder
    cluster_transforms = list(processed_dir.glob("transforms_*.json"))
    # Exclude 18 and 19
    cluster_transforms = [ct for ct in cluster_transforms if ct.stem.endswith(("_018", "_019"))]

    if (processed_dir / "transforms.json").exists():
        os.rename(processed_dir / "transforms.json", processed_dir / "transforms_all.json")
        

    for transform_path in cluster_transforms:
        cluster_id = transform_path.stem.split("_")[-1]
        print("Processing cluster:", cluster_id)
        os.rename(transform_path, processed_dir / "transforms.json")

        try:
            gs_train_command = [
                "ns-train", "splatfacto",
                "--data", processed_folder,
                "--output-dir", f"outputs/cluster_{cluster_id}",
                "--viewer.quit-on-train-completion", "True",
                "--pipeline.datamanager.cache-images", "cpu",
                "nerfstudio-data",
                "--downscale-factor", "4",
                "--orientation-method", "none",
                "--center-method", "none",
                "--auto-scale-poses", "False"
            ]

            subprocess.run(gs_train_command, cwd=data_dir)

            gs_config_path = list((data_dir / f"outputs/cluster_{cluster_id}/{processed_folder}/splatfacto").rglob("**/config.yml"))[-1]

            export_command = [
                "ns-export", "gaussian-splat",
                "--load-config", str(gs_config_path),
                "--output-dir", "exports",
                "--output-filename", f"splat_{cluster_id}.ply"
            ]

            subprocess.run(export_command, cwd=data_dir)
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")

        os.rename(processed_dir / "transforms.json", transform_path)

    os.rename(processed_dir / "transforms_all.json", processed_dir / "transforms.json")


if __name__ == "__main__":
    # compute_pointclouds()

    main()