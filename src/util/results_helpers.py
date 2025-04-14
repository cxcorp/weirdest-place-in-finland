import psycopg2
from psycopg2.extensions import connection
from pgvector.psycopg2 import register_vector
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

parquet_schema = pa.schema(
    [
        ("file_path", pa.string()),
        ("histogram", pa.list_(pa.uint8(), 96)),
        ("embedding", pa.list_(pa.float32(), 2048)),
        ("grid_x", pa.uint8()),
        ("grid_y", pa.uint8()),
    ]
)


def parquet_schema_for_embedding(embedding_size: int):
    return pa.schema(
        [
            ("file_path", pa.string()),
            ("histogram", pa.list_(pa.uint8(), 96)),
            ("embedding", pa.list_(pa.float32(), embedding_size)),
            ("grid_x", pa.uint8()),
            ("grid_y", pa.uint8()),
        ]
    )


def connect_to_db():
    print("connect to db")
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="very_secure_123",
        host="127.0.0.1",
        port=5432,
    )
    register_vector(conn)
    return conn


def dump_run_results_to_parquet(conn: connection, run_id: int, output_file: str):
    print("get data")
    with conn, conn.cursor() as curs:
        curs.execute(
            """
            SELECT file_path, histogram, embedding, grid_x, grid_y
            FROM results
            WHERE run_id = %s
            """,
            (run_id,),
        )
        rows = curs.fetchall()

    # dump rows to parquet
    print("mankel columns from rows")
    file_paths = [row[0] for row in rows]
    histograms = [row[1] for row in rows]
    embeddings = [row[2] for row in rows]
    grid_x = [row[3] for row in rows]
    grid_y = [row[4] for row in rows]

    print("create table")
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

    print("write table to parquet")
    pq.write_table(table, output_file)


def read_run_results_from_parquet(parquet_file_path: str):
    # read the parquet file into a pandas DataFrame
    table = pq.read_table(parquet_file_path, memory_map=True)
    file_paths = table["file_path"].to_pylist()
    histograms = np.stack(table["histogram"].to_numpy())
    embeddings = np.stack(table["embedding"].to_numpy())
    grid_x = np.stack(table["grid_x"].to_numpy())
    grid_y = np.stack(table["grid_y"].to_numpy())
    gridpoints = np.stack((grid_x, grid_y), axis=1)
    return (file_paths, histograms, embeddings, gridpoints)
