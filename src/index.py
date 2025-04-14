import random
import time
from itertools import batched
import os
import math

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import run_pipeline
from util.io_utils import find_all_files_with_extension_recursively
from util.results_helpers import parquet_schema

# IMAGE_DIR = "/mnt/e/mml/dataset"
IMAGE_DIR = "E:\\images\\resized"

# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 224


def list_input_files(root_dir: str) -> list[str]:
    return list(set(find_all_files_with_extension_recursively(root_dir, ".jxl")))


def basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def main():
    input_file_paths = list_input_files(IMAGE_DIR)
    assert len(input_file_paths) > 0
    print(f"Preparing to process {len(input_file_paths)} files")

    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="very_secure_123",
        host="127.0.0.1",
        port=5432,
    )
    register_vector(conn)  # so we can pass np arrays directly

    try:
        run_id: int = None
        with conn, conn.cursor() as curs:
            curs.execute("SELECT DISTINCT tile_id FROM run_processed_files")
            processed_tile_names = set([r[0] for r in curs.fetchall()])

        before = len(input_file_paths)
        input_file_paths = [
            path
            for path in input_file_paths
            if basename_no_ext(path) not in processed_tile_names
        ]
        after = len(input_file_paths)
        print(f"Filtered {before - after} files that have already been processed")
        print(f"Processing {after} files")

        with conn, conn.cursor() as curs:
            curs.execute(
                """
                INSERT INTO runs (
                    input_image_count,
                    batch_size_resnet,
                    grid_size,
                    tile_size,

                    start_time
                ) VALUES (%s, %s, %s, %s, now())
                RETURNING run_id
                """,
                (len(input_file_paths), BATCH_SIZE_RESNET, GRID_SIZE, TILE_SIZE),
            )
            (run_id,) = curs.fetchone()

        os.makedirs(".\\parquets", exist_ok=True)
        parquet_path = os.path.abspath(
            os.path.join(".\\parquets", f"run-{run_id}.parquet")
        )

        # input_file_paths = random.sample(input_file_paths, 10)

        writer = pq.ParquetWriter(parquet_path, parquet_schema)

        try:

            CHUNK_SIZE = math.ceil(
                1000 / BATCH_SIZE_RESNET
            )  # approx 1000 rows per chunk
            start = time.perf_counter()
            pipeline = run_pipeline(
                input_file_paths, batch_size_resnet=BATCH_SIZE_RESNET
            )
            for chunk in batched(pipeline, CHUNK_SIZE):
                with conn, conn.cursor() as curs:
                    execute_values(
                        curs,
                        """
                        INSERT INTO run_processed_files (run_id, tile_id, file_path) VALUES %s
                        ON CONFLICT (tile_id, run_id) DO NOTHING
                        """,
                        [(run_id, basename_no_ext(path), path) for path, _ in chunk],
                    )

                    rows = [
                        (
                            label,
                            histogram.flatten(),
                            embedding,
                            point["x"],
                            point["y"],
                        )
                        for label, batch in chunk
                        for point, embedding, histogram in batch
                    ]
                    file_paths = [row[0] for row in rows]
                    histograms = [row[1] for row in rows]
                    embeddings = [row[2] for row in rows]
                    grid_x = [row[3] for row in rows]
                    grid_y = [row[4] for row in rows]

                    table = pa.Table.from_arrays(
                        [
                            pa.array(file_paths, type=pa.string()),
                            pa.array(histograms, type=pa.list_(pa.uint8(), 96)),
                            pa.array(embeddings, type=pa.list_(pa.float32(), 2048)),
                            pa.array(grid_x, type=pa.uint8()),
                            pa.array(grid_y, type=pa.uint8()),
                        ],
                        schema=parquet_schema,
                    )
                    writer.write_table(table)
        finally:
            writer.close()
        end = time.perf_counter()
        print(
            f"Processed {len(input_file_paths)} files in {end - start:.2f} seconds ({(len(input_file_paths)/(end - start)):.2f} files/sec)"
        )

        with conn, conn.cursor() as curs:
            curs.execute(
                """
                UPDATE runs SET finish_time = now()
                WHERE run_id = %s
                """,
                (run_id,),
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
