from typing import List

import psycopg2

from pipeline import run_pipeline
from io_utils import find_all_files_with_extension_recursively

# IMAGE_DIR = "/mnt/e/mml/dataset"
IMAGE_DIR = "/mnt/e/mml/orto"

BATCH_SIZE_RESIZE = 4
# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 244


def list_input_files(root_dir: str) -> List[str]:
    return list(find_all_files_with_extension_recursively(root_dir, ".jp2"))


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
    try:
        run_id: int = None
        with conn, conn.cursor() as curs:
            curs.execute(
                """
                INSERT INTO runs (
                    input_image_count,
                    batch_size_resize,
                    batch_size_resnet,
                    grid_size,
                    tile_size,

                    start_time
                ) VALUES (%s, %s, %s, %s, %s, now())
                RETURNING run_id
                """,
                (
                    len(input_file_paths),
                    BATCH_SIZE_RESIZE,
                    BATCH_SIZE_RESNET,
                    GRID_SIZE,
                    TILE_SIZE,
                ),
            )
            (run_id,) = curs.fetchone()

        for label, embedding, coords in run_pipeline(
            input_file_paths,
            batch_size_resize=BATCH_SIZE_RESIZE,
            batch_size_resnet=BATCH_SIZE_RESNET,
            grid_size=GRID_SIZE,
            tile_size=TILE_SIZE,
            dali_num_threads=8,
        ):
            with conn, conn.cursor() as curs:
                curs.execute(
                    """
                    INSERT INTO results (run_id, file_path, grid_x, grid_y, embedding) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        label,
                        coords["x"],
                        coords["y"],
                        embedding.tolist(),
                    ),
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
