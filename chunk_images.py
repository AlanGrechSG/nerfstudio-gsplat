from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from math import cos, radians, sin
import os
from pathlib import Path
import shutil
import subprocess
from scipy.spatial.transform import Rotation as R

import numpy as np


def chunk_image_list(image_list: list[Path], cluster_size: int, overlap: int) -> list[tuple[list[Path], list[Path]]]:
    chunks = []
    total_images = len(image_list)

    num_new_images = cluster_size - overlap
    for i in range(0, total_images, num_new_images):
        end = min(i + num_new_images, total_images)
        chunk = image_list[i:end]

        overlap_next_end = min(end + overlap, total_images)
        overlap_next = [] if end >= total_images else image_list[end:overlap_next_end]

        chunks.append((chunk, overlap_next))

    return chunks

def get_chunk_name(index: int) -> str:
    return f"chunk_{index:04d}"

def get_frame_name(index: int) -> str:
    return f"frame_{index:05d}"

def copy_file(src, dst):
    """Copy a single file (wrapper so it works in a thread)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return dst  # optional, useful for debugging

def process_chunk(i, chunk, overlap_next, src_folder, cluster_size, overlap):
    futures = []
    out_dir = os.path.join(src_folder, "chunks", get_chunk_name(i))
    os.makedirs(out_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=32) as executor:
        # Copy main chunk frames
        for j, frame in enumerate(chunk):
            dest_name = f"{get_frame_name(j)}.{frame.suffix.lstrip('.')}"
            dest_path = os.path.join(src_folder, "chunks", get_chunk_name(i), dest_name)
            futures.append(executor.submit(copy_file, frame, dest_path))

        # Copy overlap frames
        for j, frame in enumerate(overlap_next):
            dest_name = f"overlap_{get_chunk_name(i + 1)}_{get_frame_name(j)}.{frame.suffix.lstrip('.')}"
            dest_path = os.path.join(src_folder, "chunks", get_chunk_name(i), dest_name)
            futures.append(executor.submit(copy_file, frame, dest_path))

        # OPTIONAL: Wait and print progress
        for f in as_completed(futures):
            _ = f.result()  # or print it

def load_camera_extrinsics_from_csv(csv_path: str) -> dict[str, np.ndarray]:
    cameras = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # --- Extrinsic parameters ---
            # Position (RealityScan: lon=x, lat=y, alt)
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["alt"])

            position = np.array([x, y, z])

            # Rotation angles (degrees → radians)
            heading = radians(float(row["heading"]))  # yaw (Z)
            pitch   = radians(float(row["pitch"]))    # pitch (Y)
            roll    = radians(float(row["roll"]))     # roll (X)

            # Build rotation matrix R = Rz * Ry * Rx
            Rz = np.array([
                [cos(heading), -sin(heading), 0],
                [sin(heading),  cos(heading), 0],
                [0,             0,            1]
            ])

            Ry = np.array([
                [cos(pitch), 0, sin(pitch)],
                [0,          1, 0],
                [-sin(pitch),0, cos(pitch)]
            ])

            Rx = np.array([
                [1, 0,          0],
                [0, cos(roll), -sin(roll)],
                [0, sin(roll),  cos(roll)]
            ])

            R = Rz @ Ry @ Rx

            # Build 4×4 extrinsic matrix [R | t]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = position

            cameras[row["#name"]] = extrinsic

    return cameras

def matrix_to_trans_rot(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    pitch = np.arcsin(-matrix[2, 0])
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])

    return x, y, z, yaw, pitch, roll

def extract_rt(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

def R_to_quat(Rm):
    return R.from_matrix(Rm).as_quat(False)  # xyzw format

def quat_to_R(q):
    return R.from_quat(q).as_matrix()

def rt_to_transform(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T

def average_quaternions(quaternions):
    """
    Average quaternions by computing the eigenvector of the summed outer product matrix.
    """
    A = np.zeros((4, 4))
    for q in quaternions:
        q = q / np.linalg.norm(q)
        A += np.outer(q, q)

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    avg_q = eigenvectors[:, np.argmax(eigenvalues)]
    return avg_q / np.linalg.norm(avg_q)

def average_translations(translations):
    return np.mean(translations, axis=0)

def average_transforms(transforms):
    rotations = []
    translations = []

    for T in transforms:
        Rm, t = extract_rt(T)
        rotations.append(R_to_quat(Rm))
        translations.append(t)

    avg_q = average_quaternions(rotations)
    avg_R = quat_to_R(avg_q)
    avg_t = average_translations(translations)

    return rt_to_transform(avg_R, avg_t)

def align_chunks(num_chunks: int, chunk_save_path: str):
    chunk_extrinsics = {}

    for i in range(num_chunks):
        cam_extr = load_camera_extrinsics_from_csv(os.path.join(chunk_save_path, get_chunk_name(i), f"{get_chunk_name(i)}_poses.csv"))
        chunk_extrinsics[i] = cam_extr

    for i in reversed(range(num_chunks - 1)):
        extr = chunk_extrinsics[i]

        overlaps_extr: dict[str, np.ndarray] = {n: e for n, e in extr.items() if n.startswith("overlap_")}

        overlap_transforms = []

        for overlap_name, overlap_extr in overlaps_extr.items():
            target_name = overlap_name.replace(f"overlap_{get_chunk_name(i + 1)}_", "")
            if target_name not in chunk_extrinsics[i + 1]:
                continue
            target_extr = chunk_extrinsics[i + 1][target_name]

            # Compute transformation to align chunk i to chunk i + 1
            transform = target_extr @ np.linalg.inv(overlap_extr)
            overlap_transforms.append(transform)

        assert len(overlap_transforms) > 0, "No overlapping frames found between chunks."
        avg_transform = average_transforms(overlap_transforms)

        for k in extr.keys():
            chunk_extrinsics[i][k] = avg_transform @ extr[k]

    combined_csv = []

    # Save updated extrinsics back to CSV
    for i in range(num_chunks):
        out_csv_path = os.path.join(chunk_save_path, get_chunk_name(i), f"{get_chunk_name(i)}_poses_aligned.csv")
        extr = chunk_extrinsics[i]

        cur_csv = []
        with open(os.path.join(chunk_save_path, get_chunk_name(i), f"{get_chunk_name(i)}_poses.csv"), 'r') as f:
            reader = csv.DictReader(f)
            cur_csv = [row for row in reader]

        with open(out_csv_path, 'w+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cur_csv[0].keys())
            writer.writeheader()
            for row in cur_csv:
                name = row["#name"]
                T = extr[name]
                x, y, z, yaw, pitch, roll = matrix_to_trans_rot(T)
                row["x"] = f"{x:.15f}"
                row["y"] = f"{y:.15f}"
                row["alt"] = f"{z:.15f}"
                row["heading"] = f"{np.degrees(yaw):.15f}"
                row["pitch"] = f"{np.degrees(pitch):.15f}"
                row["roll"] = f"{np.degrees(roll):.15f}"
                combined_csv.append({**row, "#name": os.path.join(get_chunk_name(i), name)})
                    
                writer.writerow(row)

    # Save combined CSV
    combined_csv_path = os.path.join(chunk_save_path, "aligned_poses.csv")
    with open(combined_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=combined_csv[0].keys())
        writer.writeheader()
        writer.writerows(combined_csv)

if __name__ == "__main__":
    src_folder = "D:\\st-johns_test-data\\altar"
    rs_exe = "C:\\Program Files\\Epic Games\\RealityScan_2.0\\RealityScan.exe"
    rs_save_path = os.path.join(src_folder, "RealityScanProject", "Project.rsproj")
    chunk_save_path = os.path.join(src_folder, "chunks")
    # os.makedirs(os.path.dirname(rs_save_path), exist_ok=True)

    # image_files = [
    #     Path(os.path.join(src_folder, f)) for f in os.listdir(src_folder)
    #     if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    # ]

    # cluster_size = 500
    # overlap = 100

    # chunks = chunk_image_list(image_files, cluster_size, overlap)

    # os.makedirs(chunk_save_path, exist_ok=True)

    # for i, (chunk, overlap_next) in enumerate(chunks):
    #     print("Copying files for chunk", i)
    #     process_chunk(i, chunk, overlap_next, src_folder, cluster_size, overlap)

    #     print(f"Processing chunk {i + 1} / {len(chunks)} with {len(chunk) + len(overlap_next)} images...")
    #     command = [rs_exe, "-hideUI"]
    #     # if os.path.exists(rs_save_path):
    #     #     command += ["-load", rs_save_path]

    #     command += ["-selectAllImages", "-enableAlignment", "false"]
    #     command += ["-addFolder", os.path.join(chunk_save_path, get_chunk_name(i))]
    #     command += ["-align"]
    #     command += ["-exportRegistration", os.path.join(chunk_save_path, get_chunk_name(i), f"{get_chunk_name(i)}_poses.csv"), os.path.join(src_folder, "export_params.xml")]
    #     # command += ["-save", rs_save_path]
    #     command += ["-quit"]
    #     subprocess.run(command)

    print("Done processing all chunks.")
    print("Starting merging chunks...")
    align_chunks(2, chunk_save_path)