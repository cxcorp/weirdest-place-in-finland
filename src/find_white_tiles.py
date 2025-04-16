# python modules
import os

# pip modules
from osgeo import gdal
import numpy as np
import torch
from torch.types import Device
from torchvision import tv_tensors
from tqdm import tqdm
from PIL import Image, ImageDraw
import scipy

# local modules
from util.io_utils import find_all_files_with_extension_recursively
from PIL import Image, ImageDraw, ImageFont


IMAGE_DIR = "E:\\images\\resized"

# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 224

EMBEDDING_SIZE = 512


class NoDataFilter:
    def __init__(self, threshold: int):
        # for longer strides than 255, need to use a larger datatype below at convolutions
        assert threshold <= 255
        self.threshold = threshold
        self.kernel = np.ones(threshold, dtype=np.uint8)

    def __call__(self, img: np.ndarray) -> bool:
        """
        Args:
            img_np (np.array): Input image array in HWC format with dtype uint8
        """
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8

        # create a boolean mask (N,) where each pixel is pure white (or 254), seems
        # like some images have 254 instead of 255, probably JXL compression
        white_pixels = np.all(self._extract_image_edges(img) >= 254, axis=1)

        # use 1D convolution to find runs of 'threshold' white pixels
        convolved = scipy.ndimage.convolve1d(
            white_pixels.astype(dtype=np.uint8),
            self.kernel,
            # use wrap mode so that we catch the top-left corner wraparound case
            mode="wrap",
        )

        return np.any(convolved >= self.threshold)

    def _extract_image_edges(self, img: np.ndarray) -> np.ndarray:
        top = img[0, :, :]  # (W, 3) pixels left-to-right
        bot = img[-1, :, :]  # (W, 3) pixels left-to-right
        right = img[:, -1, :]  # (H, 3) top-to-bottom
        left = img[:, 0, :]  # (H, 3) top-to-bottom

        # reverse bot and left pixel order so that bot becomes right-to-left and left becomes bottom-to-top
        # so that we have a stride of 1px in clockwise order
        bot = bot[::-1, :]
        left = left[::-1, :]

        return np.concatenate([top, right, bot, left], axis=0)


def list_input_files(root_dir: str) -> list[str]:
    return list(set(find_all_files_with_extension_recursively(root_dir, ".jxl")))


def read_image_to_tiles(path: str, device: Device = None):
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


def main():
    all_images = sorted(list_input_files(IMAGE_DIR))

    all_images = [path for path in all_images if "L2323G" in path]

    os.makedirs("./cutouts", exist_ok=True)

    has_nodata = NoDataFilter(threshold=48)

    with open("./cutouts2.txt", "w") as fp:
        for imagepath in tqdm(all_images):
            image = read_image_to_tiles(imagepath)
            # (C, H, W) -> (144, C, H/12, W/12)
            image = image.unfold(1, 224, 224).unfold(
                2, 224, 224
            )  # (3, 12, 12, 224, 224)
            image = image.permute(1, 2, 3, 4, 0).reshape(
                -1, 224, 224, 3
            )  # (144, 224, 224, 3)
            image = image.numpy()

            is_cutout = [has_nodata(image[i]) for i in range(image.shape[0])]

            if np.any(is_cutout):
                # Reshape tiles back to full image
                full_image = image.reshape(12, 12, 224, 224, 3)
                full_image = np.concatenate(
                    [np.concatenate(row, axis=1) for row in full_image], axis=0
                )

                # Convert to PIL for drawing
                pil_image = Image.fromarray(full_image.astype("uint8"))
                draw = ImageDraw.Draw(pil_image)

                # Draw red boxes on cutouts
                for i in range(len(is_cutout)):
                    if is_cutout[i]:
                        x = (i % 12) * 224
                        y = (i // 12) * 224
                        draw.rectangle([x, y, x + 224, y + 224], outline="red", width=3)
                        draw.text(
                            (x + 10, y + 10),
                            f"{i},{i % 12},{i // 12}",
                            fill="red",
                            font=ImageFont.truetype("arial.ttf", 36),
                        )

                # Resize to smaller dimensions
                pil_image = pil_image.resize((1000, 1000), Image.Resampling.BILINEAR)

                # Save image
                basename = os.path.splitext(os.path.basename(imagepath))[0]
                pil_image.save(
                    os.path.join("./cutouts", f"cutout-{basename}.jpg"), quality=75
                )

                # Write cutout locations to file
                for i in range(image.shape[0]):
                    if is_cutout[i]:
                        x = i % 12
                        y = i // 12
                        fp.write(f"{imagepath};{x};{y}\n")
    #     ax.axis("off")  # Hide axes for cleaner visualization

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
