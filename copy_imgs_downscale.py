from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Callable, Tuple

from get_image_size import get_image_size
from tqdm import tqdm
from PIL import Image


def main():
    images_dir = Path("D:\\full-cathedral\\processed")
    original_images_dir = images_dir / "images"
    image_paths = list(original_images_dir.glob("*.*"))
    downscale_factor = 4

    def copy_image(img_path, new_size_func: Callable[[Tuple[int, int]], Tuple[int, int]]):
        dest_folder = images_dir / f"images_{downscale_factor}"
        width, height = get_image_size(img_path)
        new_size = new_size_func((width, height))
        if (dest_folder / img_path.name).exists():
            try:
                target_width, target_height = get_image_size(dest_folder / img_path.name)
            except Exception:
                target_width, target_height = -1, -1

            if target_width == new_size[0] and target_height == new_size[1]:
                return

        img = Image.open(img_path)

        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        resized.save(dest_folder / img_path.name)

    transform_path = images_dir / "transforms.json"
    transforms = json.loads(transform_path.read_text())

    frame_sizes = {}
    for frame in transforms["frames"]:
        frame_sizes[Path(frame["file_path"]).name] = (frame["w"], frame["h"])
    
    with ThreadPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(copy_image, image_path, lambda size: (size[0] // downscale_factor, size[1] // downscale_factor)) for image_path in image_paths]
        for future in tqdm(as_completed(futures), "Copying images", total=len(futures), unit="image"):
            future.result()

    # for image_path in tqdm(image_paths, "Copying images", total=len(image_paths), unit="image"):        
    #     copy_image(image_path, lambda size: (size[0] // downscale_factor, size[1] // downscale_factor))

def copy_from_map():
    base_dir = Path("D:\\full-cathedral")
    processed_dir = "processed"
    original_to_ns_map_dir = base_dir / processed_dir / "original_to_ns.json"
    original_to_ns = json.loads(original_to_ns_map_dir.read_text())
    num_downscales = 5

    image_dir = base_dir / processed_dir / "images"
    downscale_dirs = [image_dir] + [base_dir / processed_dir / f"images_{2**i}" for i in range(1, num_downscales + 1)]
    for d in downscale_dirs:
        d.mkdir(exist_ok=True)

    def copy_image(img_path):
        img = Image.open(img_path)
        ns_img_name = original_to_ns[img_path.name]

        for i in range(num_downscales + 1):
            downscale_factor = 2 ** i
            new_size = (img.width // downscale_factor, img.height // downscale_factor)
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.save(downscale_dirs[i] / ns_img_name)
    
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(copy_image, base_dir / image_path) for image_path in original_to_ns.keys()]
        for future in tqdm(as_completed(futures), "Copying images", total=len(futures), unit="image"):
            future.result()

if __name__ == "__main__":
    copy_from_map()