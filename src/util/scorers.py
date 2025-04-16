import math
import sys

import numpy as np
from tqdm import trange


def score_with_LocalOutlierFactor(embeddings):
    import sklearn.neighbors

    los = sklearn.neighbors.LocalOutlierFactor(
        metric="minkowski", n_neighbors=75, contamination=0.001, n_jobs=31
    ).fit(embeddings)

    # los.negative_outlier_factor_ contains outlier scores: higher is more normal, lower is less normal
    outlier_scores = los.negative_outlier_factor_
    indices_sorted = np.argsort(outlier_scores)

    return indices_sorted, outlier_scores


def score_with_autoencoder(embeddings):
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    class VectorDataset(Dataset):
        def __init__(self, vectors):
            self.data = torch.from_numpy(vectors.astype(np.float32))
            assert self.data is not None
            print(self.data.shape)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Autoencoder: the target is the input itself.
            return self.data[idx]

    batch_size = 256
    dataset = VectorDataset(embeddings)
    del embeddings
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim=608, latent_dim=64):
            """
            input_dim : dimensionality of input data (608)
            latent_dim: dimensionality of latent space
            """
            super(Autoencoder, self).__init__()
            # Define the encoder network
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )
            # Define the decoder network
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return recon

    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=608, latent_dim=latent_dim)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    checkpoint_path = "autoencoder_maxvit_checkpoint.pth"
    best_loss = float("inf")
    model_loaded = False

    # Load checkpoint if it exists
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     best_loss = checkpoint["best_loss"]
    #     model_loaded = True
    #     print(f"Loaded checkpoint with best loss: {best_loss:.6f}")

    # Skip training if model is loaded
    if not model_loaded:
        num_epochs = 100
        pbar = trange(num_epochs)
        for epoch in pbar:
            epoch_loss = 0.0
            model.train()

            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass: get reconstruction
                recon = model(batch)

                # Compute reconstruction loss
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)
                if not math.isfinite(epoch_loss):
                    print("batch", batch)
                    print("recon", recon)
                    print("epoch_loss", epoch_loss)
                    print("loss.item()", loss.item())
                    print("batch.size(0)", batch.size(0))
                    sys.exit(1)

            epoch_loss /= len(dataset)
            if not math.isfinite(epoch_loss):
                print("epoch_loss", epoch_loss)
                print("len(dataset)", len(dataset))
                print("loss.item()", loss.item())
                print("batch.size(0)", batch.size(0))
                sys.exit(1)
            pbar.set_description(f"loss: {epoch_loss:.6f}")

            # Save checkpoint if this is the best loss so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved with best loss: {best_loss:.6f}")

    # Switch model to evaluation mode
    model.eval()
    reconstruction_errors = []

    del dataloader

    # Calculate reconstruction errors for all embeddings
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            recon = model(batch)
            errors = torch.mean((recon - batch) ** 2, dim=1)  # MSE per sample
            reconstruction_errors.extend(errors.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    # Sort indices based on reconstruction errors
    indices_sorted = np.argsort(reconstruction_errors)

    return indices_sorted, reconstruction_errors


def _calculate_variables_for_mahal(
    vector: np.ndarray, custom_mean: np.ndarray | None = None
):
    if custom_mean is None:
        mean = np.mean(vector, axis=0)
    else:
        mean = custom_mean

    covariance = np.cov(vector, rowvar=False)
    epsilon = 1e-6
    covariance += np.eye(covariance.shape[0]) * epsilon
    inv_covariance = np.linalg.inv(covariance)
    diff_to_mean = vector - mean
    return inv_covariance, diff_to_mean


def score_with_mahalanobis(
    embeddings: np.ndarray, custom_mean: np.ndarray | None = None
):
    """
    Takes a (BATCH_SIZE, 512) array and returns the argsort index of distances
    to the mean of the embeddings according to mahalanobis distance.
    """
    inv_covariance, diff_to_mean = _calculate_variables_for_mahal(
        embeddings, custom_mean
    )

    batch_size = min(embeddings.shape[0], 100_000)
    num_samples: int = embeddings.shape[0]
    embedding_mahal_distances = np.empty(num_samples, dtype=np.float32)

    for start_idx in trange(0, num_samples, batch_size, desc="Calculating distances"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_diff_to_mean = diff_to_mean[start_idx:end_idx]
        embedding_mahal_distances[start_idx:end_idx] = np.sqrt(
            np.sum(batch_diff_to_mean @ inv_covariance * batch_diff_to_mean, axis=1)
        )

    distances = embedding_mahal_distances
    indices_sorted = np.argsort(distances)

    return indices_sorted, distances


def score_with_cosine_distance(embeddings: np.ndarray, reference_vector: np.ndarray):
    """
    Takes a (BATCH_SIZE, D) array of embeddings and a reference vector of shape (D,)
    and returns the argsort index of cosine distances to the reference vector.
    """
    # Normalization is required for cosine similarity to ensure vectors are unit vectors.
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    normalized_reference = reference_vector / np.linalg.norm(reference_vector)

    # Compute cosine distances (1 - cosine similarity)
    cosine_distances = 1 - np.dot(normalized_embeddings, normalized_reference)

    # Sort indices based on cosine distances
    indices_sorted = np.argsort(cosine_distances)

    return indices_sorted
