

from sklearn.decomposition import PCA

class PCATransformer:
    def __init__(
            self,
            n_components=.98
        ):
        self.pca = PCA(n_components=n_components)

    def fit(self, x):
        self.pca.fit(x)
    def fit_and_predict(self, x):
        self.pca.fit(x)
        return self.predict(x)
    def predict(self, x, inverse=False):
        return self.pca.transform(x) if inverse == False \
            else self.pca.inverse_transform(x)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

class DimensionalityReducer:
    def __init__(self, method="pca", n_components=2, learning_rate=0.001, epochs=50, batch_size=32):
        """
        Initialize the dimensionality reduction method.
        
        Parameters:
            method (str): "pca", "tsne", "umap", or "autoencoder"
            n_components (int/float): Number of components for reduction
            learning_rate (float): Learning rate (only for Autoencoder)
            epochs (int): Number of training epochs (only for Autoencoder)
            batch_size (int): Batch size for Autoencoder training
        """
        self.method = method.lower()
        self.n_components = n_components
        self.model = None  # To store trained model
        
        if self.method == "pca":
            self.model = PCA(n_components=n_components)
        
        elif self.method == "tsne":
            self.model = TSNE(n_components=n_components, perplexity=30, random_state=42)
        
        elif self.method == "umap":
            self.model = umap.UMAP(n_components=n_components, random_state=42)
        
        elif self.method == "autoencoder":
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.autoencoder = None  # Will be initialized during fitting
        
        else:
            raise ValueError("Invalid method. Choose from 'pca', 'tsne', 'umap', or 'autoencoder'.")

    def fit(self, x):
        """
        Fit the model to the input data.
        """
        if self.method in ["pca", "tsne", "umap"]:
            self.model.fit(x)
        elif self.method == "autoencoder":
            self._train_autoencoder(x)

    def transform(self, x):
        """
        Apply dimensionality reduction on input data.
        """
        if self.method in ["pca", "umap"]:
            return self.model.transform(x)
        elif self.method == "tsne":
            return self.model.fit_transform(x)  # t-SNE does not support transform
        elif self.method == "autoencoder":
            return self._encode_autoencoder(x)

    def fit_and_transform(self, x):
        """
        Fit and then transform the data in one step.
        """
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        """
        Recover original data from reduced representation.
        Only works for PCA and Autoencoder.
        """
        if self.method == "pca":
            return self.model.inverse_transform(x)
        elif self.method == "autoencoder":
            return self._decode_autoencoder(x)
        else:
            raise NotImplementedError("Inverse transform is only supported for PCA and Autoencoder.")

    def _train_autoencoder(self, x):
        """
        Train an autoencoder model for dimensionality reduction.
        """
        x = torch.tensor(x, dtype=torch.float32)
        input_dim = x.shape[1]

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

        self.autoencoder = Autoencoder(input_dim, self.n_components)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.autoencoder.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch = batch[0]
                optimizer.zero_grad()
                encoded, decoded = self.autoencoder(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    def _encode_autoencoder(self, x):
        """
        Apply encoding using the trained autoencoder.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            return self.autoencoder.encoder(torch.tensor(x, dtype=torch.float32)).numpy()

    def _decode_autoencoder(self, x):
        """
        Decode latent representation back to original space.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            return self.autoencoder.decoder(torch.tensor(x, dtype=torch.float32)).numpy()
