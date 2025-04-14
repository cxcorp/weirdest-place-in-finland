CREATE EXTENSION vector;

CREATE TABLE runs (
  run_id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  input_image_count BIGINT NOT NULL,
  batch_size_resnet BIGINT NOT NULL,
  grid_size   BIGINT,
  tile_size   BIGINT,

  start_time  TIMESTAMP WITH TIME ZONE NOT NULL,
  finish_time TIMESTAMP WITH TIME ZONE
);

CREATE TABLE run_processed_files (
  run_id    BIGINT NOT NULL REFERENCES runs(run_id),
  tile_id   TEXT NOT NULL,
  file_path TEXT NOT NULL,

  UNIQUE (tile_id)
);

CREATE INDEX ON run_files(run_id);
