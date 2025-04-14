from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import time
import os
from os import path

# image_dir = "/mnt/e/mml/orto/normal_color_3067/mara_v_25000_50/2023/K42/02m/1"
image_dir = "/mnt/e/mml/dataset"
# image_dir = "/data/orto/normal_color_3067/mara_v_25000_50/2023/K42/02m/1"
max_batch_size = 4

file_list_path = "./filelist.txt"


def write_filelist(image_dir: str, filelist_path: str):
    files = [
        path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if path.isfile(path.join(image_dir, f)) and f.endswith(".jp2")
    ]
    files_with_label = [[file, str(i)] for i, file in enumerate(files)]
    contents = "\n".join([" ".join(pair) for pair in files_with_label])

    with open(filelist_path, "w", encoding="utf-8") as fp:
        fp.write(contents)


write_filelist(image_dir, file_list_path)

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 244


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_list=file_list_path, random_shuffle=False)
    images = fn.decoders.image(jpegs, device="mixed")

    grid_image_size = GRID_SIZE * TILE_SIZE
    # scale to GRID_SIZE*TILE_SIZE
    scaled_images = fn.resize(
        images.gpu(),
        min_filter=types.DALIInterpType.INTERP_CUBIC,
        antialias=True,
        size=(grid_image_size, grid_image_size),
    )

    return scaled_images, labels


pipe = simple_pipeline(batch_size=max_batch_size, num_threads=8, device_id=0)

start = time.perf_counter()
pipe_out = pipe.run()
end = time.perf_counter()
elapsed_ms = (end - start) * 1000
print(pipe_out)
print(f"took {elapsed_ms}ms ({elapsed_ms/1000}s)")
print(pipe_out[1])

while not pipe.empty():
    start = time.perf_counter()
    pipe_out = pipe.run()
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(pipe_out)
    print(f"took {elapsed_ms}ms ({elapsed_ms/1000}s)")
    print(pipe_out[1])
# images, labels = pipe_out
# result = images.as_cpu()

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(24, 24))
# plt.axis("off")
# plt.imshow(result.at(0))
# plt.show()

## DALI_DISABLE_NVML=1: https://github.com/NVIDIA/DALI/issues/3878
# DALI_DISABLE_NVML=1 python3 dali.py
