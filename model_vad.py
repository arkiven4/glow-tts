import torch
import modules
from torch import nn
from torch.nn import functional as F

class VAD_CartesianEncoder(nn.Module):
    def __init__(self, feature_size=256, latent_size=1024, hidden_state=768):
        super(VAD_CartesianEncoder, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_state = hidden_state

        self.fc1_a  = modules.LinearNorm(1, feature_size)
        self.fc21_a = modules.LinearNorm(feature_size, hidden_state)
        self.fc22_a = modules.LinearNorm(feature_size, hidden_state)

        self.fc1_d  = modules.LinearNorm(1, feature_size)
        self.fc21_d = modules.LinearNorm(feature_size, hidden_state)
        self.fc22_d = modules.LinearNorm(feature_size, hidden_state)

        self.fc1_v  = modules.LinearNorm(1, feature_size)
        self.fc21_v = modules.LinearNorm(feature_size, hidden_state)
        self.fc22_v = modules.LinearNorm(feature_size, hidden_state)

        # Encoder
        self.encoder_fc1 = modules.LinearNorm(hidden_state * 3, 1536) # 768 Oke
        self.encoder_fc21 = modules.LinearNorm(1536, latent_size)
        self.encoder_fc22 = modules.LinearNorm(1536, latent_size)

        self.elu = nn.ELU()

    def encode_a(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = x.unsqueeze(1) # (bs, coordinate)
        h1 = self.elu(self.fc1_a(inputs))
        z_mu = self.fc21_a(h1)
        z_var = self.fc22_a(h1)
        return z_mu, z_var
    
    def encode_d(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = x.unsqueeze(1) # (bs, coordinate)
        h1 = self.elu(self.fc1_d(inputs))
        z_mu = self.fc21_d(h1)
        z_var = self.fc22_d(h1)
        return z_mu, z_var
    
    def encode_v(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = x.unsqueeze(1) # (bs, coordinate)
        h1 = self.elu(self.fc1_v(inputs))
        z_mu = self.fc21_v(h1)
        z_var = self.fc22_v(h1)
        return z_mu, z_var
    
    def encode(self, x):
        x = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc21(x)
        logvar = self.encoder_fc22(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        arousal_input = x[:,0] - 1 # -1 because when precessing accidentaly adding 1
        valence_input = x[:,2] - 1
        dominance_input = x[:,1] - 1

        mu_a, logvar_a = self.encode_a(arousal_input) 
        mu_d, logvar_d = self.encode_d(dominance_input)
        mu_v, logvar_v = self.encode_v(valence_input)

        embeds = torch.cat([self.reparameterize(mu_a, logvar_a), self.reparameterize(mu_d, logvar_d), self.reparameterize(mu_v, logvar_v)], 1)

        mu, logvar = self.encode(embeds)
        z = self.reparameterize(mu, logvar)

        return z