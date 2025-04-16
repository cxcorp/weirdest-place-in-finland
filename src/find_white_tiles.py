# pip modules
from osgeo import gdal
import numpy as np
import torch
from torch.types import Device
from torchvision import tv_tensors
from tqdm import tqdm
from multiprocessing import Pool
from signal import signal, SIGINT

# local modules
from util.io_utils import find_all_files_with_extension_recursively
from util.no_data_filter import NoDataFilter

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_NUM_THREADS", "4")

IMAGE_DIR = "E:\\images\\resized"

# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 224

EMBEDDING_SIZE = 512


def handler(signalnum, frame):
    raise TypeError("SIGINT received, exiting...")


# attach a signal that raises a python exception so multiprocessing tasks exit on ctrl+c
signal(SIGINT, handler)


def list_input_files(root_dir: str) -> list[str]:
    return list(set(find_all_files_with_extension_recursively(root_dir, ".jxl")))


def read_image_to_tensor(path: str, device: Device = None):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray(
        xoff=0,
        yoff=0,
        xsize=ds.RasterXSize,
        ysize=ds.RasterYSize,
        buf_xsize=ds.RasterXSize,
        buf_ysize=ds.RasterYSize,
        buf_type=gdal.GDT_Byte,
    )

    arr = torch.from_numpy(arr).contiguous()
    return tv_tensors.Image(arr, device=device)


has_nodata = NoDataFilter(threshold=48)


def get_nodata_array(imagepath: str):
    image = read_image_to_tensor(imagepath, device="cuda")
    # (C, H, W) -> (144, C, H/12, W/12)
    image = image.unfold(1, 224, 224).unfold(2, 224, 224)  # (3, 12, 12, 224, 224)
    image = image.permute(1, 2, 3, 4, 0).reshape(-1, 224, 224, 3)  # (144, 224, 224, 3)
    image = image.cpu().numpy()

    is_cutout = [has_nodata(image[i]) for i in range(image.shape[0])]
    if not np.any(is_cutout):
        return None

    coords = []
    for i in range(image.shape[0]):
        if is_cutout[i]:
            x = i % 12
            y = i // 12
            coords.append((x, y))
    return imagepath, coords


def main():
    all_images = sorted(list_input_files(IMAGE_DIR))

    # os.makedirs("./cutouts", exist_ok=True)

    has_nodata = NoDataFilter(threshold=48)

    with open("./cutouts.txt", "w") as fp, Pool(12) as pool:
        for result in tqdm(
            pool.imap_unordered(get_nodata_array, all_images, chunksize=8),
            total=len(all_images),
        ):
            if result is None:
                continue

            imagepath, coords = result
            for x, y in coords:
                fp.write(f"{imagepath};{x};{y}\n")
    #     ax.axis("off")  # Hide axes for cleaner visualization

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
