
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mkit.torch_support.tensor_utils import xy_to_tensordataset

class DNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DNNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim),
        )
    
    def forward(self, x):
        return self.fc(x)

class DNNManager:
    def __init__(self, input_dim, output_dim, batch_size, num_epochs, lr, betas):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.betas = betas

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DNNModel(self.input_dim, self.output_dim).to(self.device)

    def predict(self, x, filter=False, inverse_transformer=None):
        x = torch.tensor(x).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()
        if inverse_transformer is not None:
            pred = inverse_transformer.predict(pred, inverse=True)
        return pred if not filter else gaussian_filter(pred)
    
    def train(self, features, targets, verbose=1):
        loader = xy_to_tensordataset(features, targets, return_loader=True, batch_size=self.batch_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, betas=self.betas)

        for epoch in (tqdm(range(self.num_epochs)) if verbose == 1 else range(self.num_epochs)):
            avg_loss = 0.0

            for i, (x_batch, y_batch) in enumerate(tqdm(loader, desc=f'Epoch [{epoch+1}/{self.num_epochs}]')) if verbose == 2 else enumerate(loader):
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(loader)
            
            if verbose == 2:
                print(f"Loss: {avg_loss:.4f}")
