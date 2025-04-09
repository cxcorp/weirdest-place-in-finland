CREATE EXTENSION vector;

CREATE TABLE runs (
  run_id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  input_image_count BIGINT NOT NULL,
  batch_size_resize BIGINT NOT NULL,
  batch_size_resnet BIGINT NOT NULL,
  grid_size   BIGINT NOT NULL,
  tile_size   BIGINT NOT NULL,

  start_time  TIMESTAMP WITH TIME ZONE NOT NULL,
  finish_time TIMESTAMP WITH TIME ZONE
);

CREATE TABLE results (
  run_id    BIGINT NOT NULL REFERENCES runs(run_id),
  file_path TEXT NOT NULL,
  grid_x    BIGINT NOT NULL,
  grid_y    BIGINT NOT NULL,
  embedding vector(2048) NOT NULL,

  UNIQUE (run_id, file_path, grid_x, grid_y)
);

-- fkey reverse index
CREATE INDEX ON results(run_id);