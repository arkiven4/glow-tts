import torch
import modules
from torch import nn
from torch.nn import functional as F

class VAE_Cartesian(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=512, latent_dim=1024):
        super().__init__()

        self.fcembed_1 = modules.LinearNorm(1, 16)
        self.fcembed_2 = modules.LinearNorm(1, 16)
        self.fcembed_3 = modules.LinearNorm(1, 16)

        # Encoder
        self.encoder_fc1 = modules.LinearNorm(input_dim, hidden_dim)
        self.encoder_fc21 = modules.LinearNorm(hidden_dim, latent_dim)
        self.encoder_fc22 = modules.LinearNorm(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = modules.LinearNorm(latent_dim, hidden_dim)
        self.decoder_fc2 = modules.LinearNorm(hidden_dim, 3)

        # Cartesian
        self.cartesian_fc = modules.LinearNorm(latent_dim, 3)

    def encode(self, x):
        x = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc21(x)
        logvar = self.encoder_fc22(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.decoder_fc1(z))
        x_hat = torch.sigmoid(self.decoder_fc2(z))
        return x_hat
    
    def to_cartesian(self, z):
        x_hat = torch.sigmoid(self.cartesian_fc(z))
        return x_hat

    def forward(self, x):
        embeds_x = self.fcembed_1(x[:, 0].unsqueeze(1))
        embeds_y = self.fcembed_2(x[:, 1].unsqueeze(1))
        embeds_z = self.fcembed_3(x[:, 2].unsqueeze(1))
        embeds = torch.cat([embeds_x, embeds_y, embeds_z], 1)

        mu, logvar = self.encode(embeds)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar