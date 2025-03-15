import copy
import torch
from torch import nn, optim
from tqdm import tqdm

# ----------------------------
# Conditional Segmentation VAE (No real mask input in forward)
# ----------------------------
class ConditionalSegmentationVAE(nn.Module):
    def __init__(self, latent_dim, width, height, img_channels=1, feature_maps=64, device=None):
        """
        Conditional VAE for segmentation.
        
        During training:
          - The encoder takes only the condition image and outputs latent parameters.
          - The decoder takes the condition image (encoded separately) and a latent vector z 
            (sampled via the reparameterization trick) to produce a segmentation mask.
            
        During inference:
          - Only the condition image is needed; z is sampled from a standard normal.
          
        The ground truth segmentation mask is used only in the loss computation.
        """
        super(ConditionalSegmentationVAE, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Encoder for the latent space: now takes only the condition image.
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, kernel_size=3, stride=2, padding=1),  # (feature_maps, H/2, W/2)
            nn.BatchNorm2d(feature_maps, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),  # (feature_maps*2, H/4, W/4)
            nn.BatchNorm2d(feature_maps * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc_channels = feature_maps * 2
        self.enc_h = height // 4
        self.enc_w = width // 4
        self.enc_dim = self.enc_channels * self.enc_h * self.enc_w
        
        # Fully connected layers for latent parameters:
        self.fc_mu = nn.Linear(self.enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, latent_dim)
        
        # 2. Image encoder for conditioning (used by the decoder)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, kernel_size=3, stride=2, padding=1),  # (feature_maps, H/2, W/2)
            nn.BatchNorm2d(feature_maps, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),  # (feature_maps*2, H/4, W/4)
            nn.BatchNorm2d(feature_maps * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc_img_channels = feature_maps * 2
        self.enc_img_h = height // 4
        self.enc_img_w = width // 4
        self.enc_img_dim = self.enc_img_channels * self.enc_img_h * self.enc_img_w
        
        # 3. Decoder:
        # We combine the flattened image encoding (from the condition image) with the latent vector z.
        # The fully connected layer projects the concatenated vector to an initial feature map.
        self.init_h = height // 16
        self.init_w = width // 16
        self.init_channels = feature_maps * 8  # adjust as needed
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim + self.enc_img_dim, self.init_channels * self.init_h * self.init_w),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            # Upsample 1: (init_channels) --> (feature_maps*4)
            nn.ConvTranspose2d(self.init_channels, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4, affine=False),
            nn.LeakyReLU(),
            # Upsample 2: (feature_maps*4) --> (feature_maps*2)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2, affine=False),
            nn.LeakyReLU(),
            # Upsample 3: (feature_maps*2) --> (feature_maps)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps, affine=False),
            nn.LeakyReLU(),
            # Upsample 4: (feature_maps) --> (1, full resolution)
            nn.ConvTranspose2d(feature_maps, 1, kernel_size=4, stride=2, padding=1)
        )
        
        self.to(self.device)
    
    def reparameterize(self, mu, logvar):
        """Performs the reparameterization trick to sample z."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, img):
        """
        img: condition image tensor of shape (batch, img_channels, height, width)
        
        Returns:
          seg_mask_flat: generated segmentation mask, flattened to (batch, width*height)
          mu: latent mean vector
          logvar: latent log-variance vector
        """
        batch_size, _, height, width = img.shape
        
        # --- Encoder ---
        enc_out = self.encoder(img)  # (batch, enc_channels, H/4, W/4)
        enc_flat = enc_out.view(batch_size, -1)
        mu = self.fc_mu(enc_flat)
        logvar = self.fc_logvar(enc_flat)
        z = self.reparameterize(mu, logvar)
        
        # --- Decoder ---
        img_enc = self.img_encoder(img)  # (batch, enc_img_channels, H/4, W/4)
        img_enc_flat = img_enc.view(batch_size, -1)
        combined = torch.cat([img_enc_flat, z], dim=1)
        fc_dec_out = self.fc_dec(combined)
        fc_dec_out = fc_dec_out.view(batch_size, self.init_channels, self.init_h, self.init_w)
        seg_mask = self.decoder(fc_dec_out)  # (batch, 1, height, width)
        seg_mask_flat = seg_mask.view(batch_size, -1)
        
        return seg_mask_flat, mu, logvar

    def inference(self, img):
        """
        Generates a segmentation mask from a condition image.
        
        Parameters:
          img: condition image tensor of shape (batch, img_channels, height, width)
        
        Returns:
          seg_mask_flat: the generated segmentation mask, flattened to (batch, width*height)
        """
        batch_size, _, height, width = img.shape
        img_enc = self.img_encoder(img)
        img_enc_flat = img_enc.view(batch_size, -1)
        z = torch.randn(batch_size, self.fc_mu.out_features, device=img.device)
        combined = torch.cat([img_enc_flat, z], dim=1)
        fc_dec_out = self.fc_dec(combined)
        fc_dec_out = fc_dec_out.view(batch_size, self.init_channels, self.init_h, self.init_w)
        seg_mask = self.decoder(fc_dec_out)
        seg_mask_flat = seg_mask.view(batch_size, -1)
        return seg_mask_flat

    def validate(self, val_loader, seg_criterion, kl_weight=0.001, device=None):
        """
        Evaluates the model on the validation set.
        """
        device = device or self.device
        self.eval()  # Set model to evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for imgs, true_masks in val_loader:
                imgs = imgs.to(device)
                true_masks = true_masks.to(device)
                seg_logits, mu, logvar = self(imgs)  # Only the image is passed in
                recon_loss = seg_criterion(seg_logits, true_masks)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
                loss = recon_loss + kl_weight * kl_loss
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train_vae(self, train_loader, val_loader, n_epochs, seg_criterion, kl_weight=0.001, patience=10, device=None):
        """
        Trains the conditional segmentation VAE with early stopping.
        
        Training stops if the validation loss does not improve for 'patience' consecutive epochs.
        Returns the model with the best validation performance.
        """
        device = device or self.device
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = copy.deepcopy(self.state_dict())
        
        for epoch in range(n_epochs):
            self.train()  # Set model to training mode
            total_loss = 0.0
            for imgs, true_masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                imgs = imgs.to(device)
                true_masks = true_masks.to(device)
                optimizer.zero_grad()
                seg_logits, mu, logvar = self(imgs)  # Only the condition image is input
                recon_loss = seg_criterion(seg_logits, true_masks)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
                loss = recon_loss + kl_weight * kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation step
            val_loss = self.validate(val_loader, seg_criterion, kl_weight, device)
            print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
                print(f"Validation loss improved to {best_val_loss:.4f}.")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        
        self.load_state_dict(best_model_weights)
        return self