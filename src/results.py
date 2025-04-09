from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import json

BATCH_SIZE_RESIZE = 4
# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 244


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


def main():
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="very_secure_123",
        host="127.0.0.1",
        port=5432,
    )
    try:
        with conn, conn.cursor() as curs:
            curs.execute(
                """
                SELECT file_path, grid_x, grid_y, embedding
                FROM results
                WHERE run_id = 2
                """
            )
            rows = curs.fetchall()
        metadatas = [(r[0], r[1], r[2]) for r in rows]
        all_embeddings = [json.loads(row[3]) for row in rows]
        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        print(all_embeddings.shape)  # prints (4608, 2048)

        most_usual_index, most_unusual_index = (
            most_usual_and_usual_according_to_Mahalanobis_dist(all_embeddings)
        )
        most_usual = metadatas[most_usual_index]
        most_unusual = metadatas[most_unusual_index]
        print("Most usual: ", most_usual)
        print("Most unusual: ", most_unusual)

        # clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, metric='cosine')
        # y_pred = clf.fit_predict(all_embeddings)
        # X_scores = clf.negative_outlier_factor_
        # print("Yay")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
