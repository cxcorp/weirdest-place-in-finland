# only import what we need here for get_multiple_rois_from_image
# so multiprocessing pool doesn't explode the memory
import numpy as np
from osgeo import gdal
from collections.abc import Iterable, Sequence

gdal.UseExceptions()

# Image.MAX_IMAGE_PIXELS = 12000 * 12000  # disable decompression bomb warning

BATCH_SIZE_RESIZE = 4
# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 224

EMBEDDING_SIZE = 512

PARQUETS_PATH = "./parquets/maxvit"
IGNORED_TILES_PATH = "./cutouts.txt"
RESULTS_PATH = "./results/ALL-maxvit-hybrid-mahal plus autoencoder"


def most_usual_and_usual_according_to_euclidean_dist(all_embeddings):
    mean_embedding = np.mean(all_embeddings, axis=0)
    distances = np.linalg.norm(all_embeddings - mean_embedding, axis=1)

    most_usual_index = np.argmin(distances)
    most_unusual_index = np.argmax(distances)
    return most_usual_index, most_unusual_index


def most_usual_and_usual_according_to_Mahalanobis_dist(all_embeddings):
    mean_embedding = np.mean(all_embeddings, axis=0)
    covariance = np.cov(all_embeddings, rowvar=False)

    epsilon = 1e-6
    covariance += np.eye(covariance.shape[0]) * epsilon

    inv_covariance = np.linalg.inv(covariance)
    diff = all_embeddings - mean_embedding
    mahal_distances = np.sqrt(np.sum(diff @ inv_covariance * diff, axis=1))

    most_usual_index = np.argmin(mahal_distances)
    most_unusual_index = np.argmax(mahal_distances)

    return most_usual_index, most_unusual_index


def normalize_histogram(h: list):
    h = np.array(h).astype(np.float32)
    total = h.sum()
    if total == 0:
        return h
    h = h / total  # Normalize to 0..1
    return (h * 2) - 1  # Scale to -1..1 like the maxvit embeddings


def chi_square_distance(h1, h2, eps=1e-10):
    return np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))


def get_multiple_rois_from_image(
    file_path: str, grid_size: int, rois: Iterable[tuple[int, int]]
):
    with gdal.Open(file_path) as d:
        assert d.RasterXSize == d.RasterYSize
        assert d.RasterXSize == 12 * TILE_SIZE
        arr = d.ReadAsArray(
            xoff=0,
            yoff=0,
            xsize=d.RasterXSize,
            ysize=d.RasterYSize,
            buf_xsize=d.RasterXSize,
            buf_ysize=d.RasterYSize,
            buf_type=gdal.GDT_Byte,
        )
    # (C, H, W) -> (H, W, C)
    im = arr.transpose(1, 2, 0)

    height, width, channels = im.shape
    assert (height, width) == (12 * TILE_SIZE, 12 * TILE_SIZE) and channels == 3
    assert (height % grid_size) == 0

    px_grid_size = height // grid_size

    result = dict()
    for grid_x, grid_y in rois:
        start_x = int(grid_x) * px_grid_size
        start_y = int(grid_y) * px_grid_size

        roi = im[
            start_y : start_y + px_grid_size,
            start_x : start_x + px_grid_size,
        ]

        result[(grid_x, grid_y)] = roi

    return file_path, result


def get_multiple_rois_from_image_mp(args: tuple[str, int, Iterable[tuple[int, int]]]):
    return get_multiple_rois_from_image(*args)


def get_image_roi(file_path: str, grid_x: str, grid_y: str, grid_size: int):
    im = cv2.imread(file_path)
    cv2.cvtColor(im, cv2.COLOR_BGR2RGB, dst=im)

    height, width, channels = im.shape
    assert (height, width) == (12 * TILE_SIZE, 12 * TILE_SIZE) and channels == 3

    assert (height % grid_size) == 0
    effective_grid_size = height // grid_size

    start_x = effective_grid_size * grid_x
    start_y = effective_grid_size * grid_y

    return im[
        start_y : start_y + effective_grid_size, start_x : start_x + effective_grid_size
    ]


def get_image(file_path: str):
    im = cv2.imread(file_path)
    cv2.cvtColor(im, cv2.COLOR_BGR2RGB, dst=im)

    height, width, channels = im.shape
    assert (height, width) == (TILE_SIZE, TILE_SIZE) and channels == 3

    return im


def dump_to_parquet(run_id: int):
    print("connect to db")
    conn = connect_to_db()
    try:
        dump_run_results_to_parquet(conn, run_id, f"./run_{run_id}.parquet")
    finally:
        conn.close()


def read_from_parquet(run_id: int):
    return read_run_results_from_parquet(f"./run_{run_id}.parquet")


class IgnoredTiles:
    def __init__(self, ignored_tiles_file_path: str):
        # for longer strides than 255, need to use a larger datatype below at convolutions
        self.ignored = self._read_ignored_tiles_from_file(ignored_tiles_file_path)

    def is_ignored(self, file_path: str, gridpoint: Sequence[int]):
        x = gridpoint[0]
        y = gridpoint[1]

        if file_path not in self.ignored:
            return False

        return (x, y) in self.ignored[file_path]

    def _read_ignored_tiles_from_file(
        self,
        ignore_file_path: str,
    ) -> dict[str, set[tuple[int, int]]]:
        ignored = dict()

        with open(ignore_file_path, "r") as fp:
            all_lines = fp.readlines()
            all_lines = [l.strip() for l in all_lines if len(l.strip()) > 0]
            for line in all_lines:
                file_path, x_str, y_str = line.split(";")

                ignored_tiles = ignored.setdefault(file_path, set())
                ignored_tiles.add((int(x_str), int(y_str)))

        return ignored


def read_from_parquet_dataset(parquets_path: str):
    dataset = ds.dataset(parquets_path, format="parquet")
    rowcount: int = dataset.count_rows()

    file_paths: list[str] = list()
    histograms = np.empty((rowcount, 96), dtype=np.uint8)
    embeddings = np.empty((rowcount, EMBEDDING_SIZE), dtype=np.float32)
    grid_x = np.empty((rowcount,), dtype=np.uint8)
    grid_y = np.empty((rowcount,), dtype=np.uint8)

    i = 0
    pbar = tqdm(total=rowcount, desc="Reading parquets")
    for batch in dataset.to_batches():
        batch_embeddings = np.stack(batch["embedding"].to_numpy(zero_copy_only=False))

        # filter out NaN embeddings
        non_nan_indices = np.argwhere(
            ~np.any(np.isnan(batch_embeddings), axis=1)
        ).flatten()

        num_valid_rows = non_nan_indices.shape[0]

        start_idx = i
        end_idx = i + num_valid_rows

        embeddings[start_idx:end_idx] = batch_embeddings[non_nan_indices]
        histograms[start_idx:end_idx] = np.stack(
            batch["histogram"].to_numpy(zero_copy_only=False)
        )[non_nan_indices]
        grid_x[start_idx:end_idx] = np.stack(
            batch["grid_x"].to_numpy(zero_copy_only=False)
        )[non_nan_indices]
        grid_y[start_idx:end_idx] = np.stack(
            batch["grid_y"].to_numpy(zero_copy_only=False)
        )[non_nan_indices]

        batch_file_paths = batch["file_path"].to_pylist()

        file_paths.extend([batch_file_paths[i] for i in non_nan_indices])

        i += num_valid_rows
        pbar.update(num_valid_rows)

    pbar.close()

    # Shrink the arrays to fit the filtered rows
    # refcheck=True (the default) will cause an error here if debugger is attached
    histograms.resize((i, histograms.shape[1]))
    embeddings.resize((i, embeddings.shape[1]))
    grid_x.resize((i,))
    grid_y.resize((i,))

    gridpoints = np.stack((grid_x, grid_y), axis=1)

    assert embeddings.shape[0] == i

    print(f"Read {i} rows from parquet (filtered out {rowcount-i})")
    return file_paths, histograms, embeddings, gridpoints


def main():
    print("read")
    start_time = time.perf_counter()
    file_paths, histograms, embeddings, gridpoints = read_from_parquet_dataset(
        PARQUETS_PATH
    )
    end_time = time.perf_counter()
    print(f"read took {end_time - start_time:.2f} seconds")

    print("read ignored tiles")
    ignored_tiles = IgnoredTiles("./cutouts.txt")

    print("filter away empty images")
    valid_indices = np.array(
        [
            i
            for i in range(len(file_paths))
            if not ignored_tiles.is_ignored(file_paths[i], gridpoints[i])
        ]
    )
    print(f"  valid:   {len(valid_indices)}")
    print(
        f"  invalid: {len(file_paths) - len(valid_indices)} ({((len(file_paths) - len(valid_indices)) / len(file_paths) * 100):.1f}%)"
    )

    file_paths = [file_paths[i] for i in valid_indices]
    valid_embeddings = embeddings[valid_indices].copy()
    del embeddings
    valid_gridpoints = gridpoints[valid_indices].copy()
    del gridpoints

    print("normalize histograms 0..255 -> -1..1")
    histograms_normalized = np.array(
        list(
            tqdm(
                (normalize_histogram(h) for h in histograms[valid_indices]),
                total=len(histograms[valid_indices]),
            )
        )
    )
    del histograms

    plot_ääripäät(valid_embeddings, histograms_normalized, file_paths, valid_gridpoints)


def scale_to_0_1(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def plot_ääripäät(embeddings, histograms, paths, gridpoints):
    _, mahalanobis_distances = scorers.score_with_mahalanobis(embeddings)
    concated = np.concatenate((embeddings, histograms), axis=1)
    del embeddings
    del histograms
    _, autoencoder_distances = scorers.score_with_autoencoder(concated)

    # scale distances to 0..1
    mahalanobis_distances = scale_to_0_1(mahalanobis_distances)
    autoencoder_distances = scale_to_0_1(autoencoder_distances)

    # Sum the distances
    combined_distances = mahalanobis_distances + autoencoder_distances
    indices_sorted = np.argsort(combined_distances)

    IMAGES_PER_GROUP_TO_SHOW = 300

    most_usual_indices = indices_sorted[:IMAGES_PER_GROUP_TO_SHOW]
    most_unusual_indices = indices_sorted[-IMAGES_PER_GROUP_TO_SHOW:][::-1]

    # read images en bulk with get_multiple_rois_from_image()
    image_path_to_rois = dict()
    for i in most_usual_indices:
        grid_x, grid_y = gridpoints[i]
        filepath = paths[i]
        if filepath not in image_path_to_rois:
            image_path_to_rois[filepath] = set()
        image_path_to_rois[filepath].add((grid_x, grid_y))
    for i in most_unusual_indices:
        grid_x, grid_y = gridpoints[i]
        filepath = paths[i]
        if filepath not in image_path_to_rois:
            image_path_to_rois[filepath] = set()
        image_path_to_rois[filepath].add((grid_x, grid_y))

    print("read images from disk")
    image_path_to_images = dict()
    with Pool(8) as pool:
        tasks = [
            (file_path, GRID_SIZE, rois)
            for file_path, rois in image_path_to_rois.items()
        ]

        for filepath, result in tqdm(
            pool.imap_unordered(get_multiple_rois_from_image_mp, tasks, 8),
            desc="Reading images",
            total=len(tasks),
        ):
            image_path_to_images[filepath] = result

    def gather_images_from_indices(indices):
        output = []
        for i in indices:
            grid_x, grid_y = gridpoints[i]
            filepath = paths[i]
            roi = image_path_to_images[filepath][(grid_x, grid_y)]

            # label is basename without extension combined with grid coordinates
            label = os.path.splitext(os.path.basename(filepath))[0]
            label += f" ({grid_x}, {grid_y})"

            output.append((label, roi))
        return output

    most_usuals = gather_images_from_indices(most_usual_indices)
    most_unusuals = gather_images_from_indices(most_unusual_indices)

    print("write results to disk")
    output_dir = RESULTS_PATH
    os.makedirs(output_dir + "/usual", exist_ok=True)
    os.makedirs(output_dir + "/unusual", exist_ok=True)
    for i, (label, img) in enumerate(most_usuals):
        cv2.imwrite(
            os.path.join(output_dir, "usual", f"{i:03}_{label}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
    for i, (label, img) in enumerate(most_unusuals):
        cv2.imwrite(
            os.path.join(output_dir, "unusual", f"{i:03}_{label}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    import os
    import time
    from multiprocessing import Pool

    import cv2
    import pyarrow.dataset as ds
    from tqdm import tqdm

    from util.results_helpers import (
        dump_run_results_to_parquet,
        read_run_results_from_parquet,
        connect_to_db,
    )

    import util.scorers as scorers

    main()
