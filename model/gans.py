import torch
import torch.nn as nn
import torch.optim as optim
from mkit.torch_support.tensor_utils import xy_to_tensordataset
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class GANManager:
    def __init__(
            self,
            noise_dim,
            condition_dim,
            output_dim,
            batch_size,
            num_epochs,
            lr,
            betas
        ):
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.betas = betas

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instantiate the generator and discriminator
        self.G = Generator(self.noise_dim, self.condition_dim, self.output_dim).to(self.device)
        self.D = Discriminator(input_dim=self.condition_dim + self.output_dim).to(self.device)
    def predict(self, x, filter=False, inverse_transformer=None):
        with torch.no_grad():
            x = torch.tensor(x).float().to(self.device)
            noise = torch.randn(len(x), self.noise_dim, device=self.device)
            pred = self.G(noise, x).detach().cpu().numpy()
        if inverse_transformer is not None:
            pred = inverse_transformer.predict(pred, inverse=True)
        return pred if not filter else gaussian_filter(pred)
        
    def train(self, features, transformed, verbose=1):
        # Device configuration

        # Create the dataloader (assumes xy_to_tensordataset returns (condition, image) pairs)
        loader = xy_to_tensordataset(features, transformed, return_loader=True, batch_size=self.batch_size)


        # Loss function and optimizers
        criterion = nn.BCELoss()
        optimizer_G = optim.Adamax(self.G.parameters(), lr=self.lr, betas=self.betas)
        optimizer_D = optim.Adamax(self.D.parameters(), lr=self.lr, betas=self.betas)
        for epoch in (tqdm(range(self.num_epochs)) if verbose == 1 else range(self.num_epochs)):
            avg_d_loss = 0.0
            avg_g_loss = 0.0

            for i, (conditions, real_images) in enumerate(tqdm(loader, desc=f'Epoch [{epoch+1}/{self.num_epochs}]')) if verbose == 2 else enumerate(loader):
                # Move real images and condition vectors to device
                real_images = real_images.to(self.device).float()
                conditions = conditions.to(self.device).float()
                current_batch_size = real_images.size(0)
                
                # Create ground-truth labels
                valid = torch.ones(current_batch_size, 1, device=self.device)  # Label 1 for real images
                fake = torch.zeros(current_batch_size, 1, device=self.device)  # Label 0 for fake images

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Generate fake images with noise and the same conditions
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                fake_images = self.G(noise, conditions)

                # Loss for real images
                real_loss = criterion(self.D(real_images, conditions), valid)
                # Loss for fake images (detach to avoid backprop into generator)
                fake_loss = criterion(self.D(fake_images.detach(), conditions), fake)
                d_loss = (real_loss + fake_loss) / 2

                # Backprop and update discriminator
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate fake images (again) to compute generator loss
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                fake_images = self.G(noise, conditions)
                # Generator loss: aim to fool the discriminator (labels are valid)
                g_loss = criterion(self.D(fake_images, conditions), valid)

                # Backprop and update generator
                g_loss.backward()
                optimizer_G.step()
                avg_g_loss += g_loss.item() / len(loader)
                avg_d_loss += d_loss.item() / len(loader)
            if verbose == 2:
                print(f"Loss D: {avg_d_loss:.4f} | Loss G: {avg_g_loss:.4f}")

# Generator: Fully connected network generating an output of the same dimension as the condition vector
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim, hidden_dim=64):
        """
        noise_dim: Dimension of the noise vector
        condition_dim: Dimension of the condition vector (also the output dimension)
        hidden_dim: Number of hidden neurons in fully connected layers
        """
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, output_dim),
        )
    
    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        return self.fc(x)

# Discriminator: Fully connected network assessing validity of the input data
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        condition_dim: Dimension of the condition vector (same as the input/output dimension)
        hidden_dim: Number of hidden neurons in fully connected layers
        """
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input includes both generated data and condition
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()  # Output probability of real/fake
        )
    
    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        return self.fc(x)

