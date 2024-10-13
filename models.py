import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import modules_gst
import modules_vits
import commons
import attentions
import monotonic_align
from model_emocatch import EmoCatcher

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

# class VAD_CartesianEncoder(nn.Module):
#     def __init__(self, hidden_state=96, latent_size=256):
#         super(VAD_CartesianEncoder, self).__init__()
#         self.latent_size = latent_size
#         self.hidden_state = hidden_state

#         self.emb_a = modules.LinearNorm(1, hidden_state)
#         self.emb_v = modules.LinearNorm(1, hidden_state)
#         self.emb_d = modules.LinearNorm(1, hidden_state)

#         self.emb_style = modules.LinearNorm(hidden_state * 3, latent_size)
#         self.emotion_linear = nn.Sequential(nn.Linear(latent_size, latent_size), nn.ReLU())

#     def forward(self, x):
#         arousal_input = self.emb_a(x[:,0].unsqueeze(1) - 1) # -1 because when precessing accidentaly adding 1
#         valence_input = self.emb_v(x[:,2].unsqueeze(1) - 1)
#         dominance_input = self.emb_d(x[:,1].unsqueeze(1) - 1)

#         embeds_cat = torch.cat([arousal_input, valence_input, dominance_input], 1)
#         embeds_cat = self.emotion_linear(self.emb_style(embeds_cat))
#         return embeds_cat
    
# class VAD_CartesianEncoderVAE(nn.Module):
#     def __init__(self, hidden_state=96, latent_size=256):
#         super(VAD_CartesianEncoderVAE, self).__init__()
#         self.latent_size = latent_size
#         self.hidden_state = hidden_state

#         self.emb_a = modules.LinearNorm(1, hidden_state)
#         self.emb_v = modules.LinearNorm(1, hidden_state)
#         self.emb_d = modules.LinearNorm(1, hidden_state)

#         # Encoder
#         self.encoder_fc1 = modules.LinearNorm(hidden_state * 3, latent_size * 2)
#         self.encoder_fc21 = modules.LinearNorm(latent_size * 2, latent_size)
#         self.encoder_fc22 = modules.LinearNorm(latent_size * 2, latent_size)

#     def encode(self, x):
#         x = F.relu(self.encoder_fc1(x))
#         mu = self.encoder_fc21(x)
#         logvar = self.encoder_fc22(x)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std
    
#     def forward(self, x):
#         arousal_input = self.emb_a(x[:,0].unsqueeze(1) - 1) # -1 because when precessing accidentaly adding 1
#         valence_input = self.emb_v(x[:,2].unsqueeze(1) - 1)
#         dominance_input = self.emb_d(x[:,1].unsqueeze(1) - 1)

#         embeds_cat = torch.cat([arousal_input, valence_input, dominance_input], 1)
#         mu, logvar = self.encode(embeds_cat)
#         z = self.reparameterize(mu, logvar)
#         return z, mu
    
class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''
    def __init__(self, n_mel_channels=80,
          style_hidden=256,
          style_vector_dim=512,
          style_kernel_size=5,
          style_head=2,
          dropout=0.1):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel_channels 
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            modules_vits.LinearNorm(self.in_dim, self.hidden_dim),
            modules_vits.Mish(),
            nn.Dropout(self.dropout),
            modules_vits.LinearNorm(self.hidden_dim, self.hidden_dim),
            modules_vits.Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            modules_vits.Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            modules_vits.Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = modules_vits.MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = modules_vits.LinearNorm(self.hidden_dim, self.out_dim)

        # self.mu = nn.Linear(self.out_dim, 256)
        # self.log_var = nn.Linear(self.out_dim, 256)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std

    def forward(self, x):
        #print(x.shape)
        x = x.transpose(2, 1)
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1,2)
        x = self.temporal(x)
        x = x.transpose(1,2)
        
        x, _ = self.slf_attn(x, mask=None)
        x = self.fc(x)
        w = self.temporal_avg_pool(x, mask=None).unsqueeze(2)

        # mu = self.mu(w)
        # logvar = self.log_var(w)
        # z = self.reparameterize(mu, logvar).unsqueeze(2)

        return w

#Interference Example https://github.com/jinhan/tacotron2-gst/blob/master/inference.ipynb
class GST(nn.Module):
    def __init__(self, token_num, token_embedding_size, num_heads, ref_enc_filters, n_mel_channels, ref_enc_gru_size, gin_channels=0, lin_channels=0):
        super().__init__()
        #self.encoder = modules_gst.ReferenceEncoderNew(ref_enc_filters, n_mel_channels, ref_enc_gru_size)
        self.encoder = modules_gst.ReferenceEncoder(ref_enc_filters, n_mel_channels, ref_enc_gru_size)
        self.stl = modules_gst.STL(token_num, token_embedding_size, num_heads, ref_enc_gru_size)

        # self.mu = nn.Linear(ref_enc_gru_size, 32)
        # self.log_var = nn.Linear(ref_enc_gru_size, 32)
        # self.fc3 = nn.Linear(32, 256)

        if gin_channels != 0:
          self.cond = nn.Conv1d(gin_channels, ref_enc_gru_size, 1)
          
        if lin_channels != 0:
          self.cond_l = nn.Conv1d(lin_channels, ref_enc_gru_size, 1)

    # def encode_param(self, x):
    #     x = F.relu(self.encoder_fc1(x))
    #     mu = self.mu(x)
    #     logvar = self.log_var(x)
    #     return mu, logvar

    def reparameterize(self, mu, logvar, infer):
        # if infer == False:

        # else:
        #     return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, inputs, input_lengths=None, infer=False, g=None, l=None):
        enc_out = self.encoder(inputs)
        #enc_out = self.encoder(inputs, input_lengths=input_lengths)
        # if g is not None:
        #   g = torch.detach(g)
        #   enc_out = enc_out + self.cond(g).squeeze(-1)

        # if l is not None:
        #   l = torch.detach(l)
        #   enc_out = enc_out + self.cond_l(l).squeeze(-1)

        # mu = self.mu(enc_out)
        # logvar = self.log_var(enc_out)
        # z = self.reparameterize(mu, logvar, infer)
        # style_embed = self.fc3(z).unsqueeze(2)

        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        style_embed = self.stl(enc_out).transpose(1,2)
        return style_embed, None

class GSTNoReff(nn.Module):
    def __init__(self, token_num, token_embedding_size, num_heads, ref_enc_filters, n_mel_channels, ref_enc_gru_size, emoin_channels=0, lin_channels=0):
        super().__init__()

        if emoin_channels != 0:
          self.cond_emo = nn.Linear(emoin_channels, ref_enc_gru_size)

        self.stl = modules_gst.STL(token_num, token_embedding_size, num_heads, ref_enc_gru_size)

    def forward(self, inputs):
        enc_out = self.cond_emo(inputs) # torch.Size([12, 128])
        style_embed = self.stl(enc_out).transpose(1,2)

        return style_embed

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0, lin_channels=0, emoin_channels=0):
    super().__init__()
    # if lin_channels != 0 :
    #   in_channels += lin_channels
    
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels

    # condition encoder text
    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.convs = modules.DilatedDepthSeparableConv(filter_channels, kernel_size, num_layers=3, dropout_p=p_dropout)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

    # posterior encoder
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    self.flows += [modules.ConvFlow(2, filter_channels, kernel_size, num_layers=3) for _ in range(n_flows)]

    # condition encoder duration
    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_convs = modules.DilatedDepthSeparableConv(filter_channels, kernel_size, num_layers=3, dropout_p=p_dropout)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)

    # flow layers
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    self.post_flows += [modules.ConvFlow(2, filter_channels, kernel_size, num_layers=3) for _ in range(n_flows)]

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    if lin_channels != 0 :
      self.cond_lang = nn.Conv1d(lin_channels, filter_channels, 1)

    if emoin_channels != 0:
      self.cond_emo = nn.Conv1d(emoin_channels, filter_channels, 1)

  def forward(self, x, x_mask, dr=None, g=None, l=None, emo=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)

    if g is not None:
        g = torch.detach(g)
        x = x + self.cond(g)

    if emo is not None:
        emo = torch.detach(emo)
        x = x + self.cond_emo(emo)

    if l is not None:
        l = torch.detach(l)
        x = x + self.cond_lang(l)

    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
        flows = self.flows
        assert dr is not None

        # condition encoder duration
        h = self.post_pre(dr)
        h = self.post_convs(h, x_mask)
        h = self.post_proj(h) * x_mask
        noise = torch.randn(dr.size(0), 2, dr.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
        z_q = noise

        # posterior encoder
        logdet_tot_q = 0.0
        for idx, flow in enumerate(self.post_flows):
            z_q, logdet_q = flow(z_q, x_mask, g=(x + h))
            logdet_tot_q = logdet_tot_q + logdet_q
            if idx > 0:
                z_q = torch.flip(z_q, [1])

        z_u, z_v = torch.split(z_q, [1, 1], 1)
        u = torch.sigmoid(z_u) * x_mask
        z0 = (dr - u) * x_mask

        # posterior encoder - neg log likelihood
        logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
        nll_posterior_encoder = (
            torch.sum(-0.5 * (math.log(2 * math.pi) + (noise**2)) * x_mask, [1, 2]) - logdet_tot_q
        )

        z0 = torch.log(torch.clamp_min(z0, 1e-5)) * x_mask
        logdet_tot = torch.sum(-z0, [1, 2])
        z = torch.cat([z0, z_v], 1)

        # flow layers
        for idx, flow in enumerate(flows):
            z, logdet = flow(z, x_mask, g=x, reverse=reverse)
            logdet_tot = logdet_tot + logdet
            if idx > 0:
                z = torch.flip(z, [1])

        # flow layers - neg log likelihood
        nll_flow_layers = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
        return nll_flow_layers + nll_posterior_encoder

    flows = list(reversed(self.flows))
    flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
    z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
    for flow in flows:
        z = torch.flip(z, [1])
        z = flow(z, x_mask, g=x, reverse=reverse)

    z0, _ = torch.split(z, [1, 1], 1)
    logw = z0
    return logw
  
class StochasticPitchPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0, emoin_channels=0):
    super().__init__()
    
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.emoin_channels = emoin_channels

    # condition encoder text
    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.convs = modules.DilatedDepthSeparableConv(filter_channels, kernel_size, num_layers=3, dropout_p=p_dropout)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

    # posterior encoder
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    self.flows += [modules.ConvFlow(2, filter_channels, kernel_size, num_layers=3) for _ in range(n_flows)]

    # if gin_channels != 0:
    #   self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    if emoin_channels != 0:
      self.cond_emo = nn.Conv1d(emoin_channels, filter_channels, 1)

  def forward(self, x, x_mask, dr=None, g=None, emo=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)

    # if g is not None:
    #     g = torch.detach(g)
    #     x = x + self.cond(g)

    if emo is not None:
        emo = torch.detach(emo)
        x = x + self.cond_emo(emo)

    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
        flows = self.flows
        assert dr is not None

        noise = torch.randn(dr.size()).to(device=x.device, dtype=x.dtype) * x_mask
        z = torch.cat([dr, noise], 1)

        logdet_tot = 0
        # flow layers
        for idx, flow in enumerate(flows):
            z, logdet = flow(z, x_mask, g=x, reverse=reverse)
            logdet_tot = logdet_tot + logdet
            if idx > 0:
                z = torch.flip(z, [1])

        # flow layers - neg log likelihood
        nll_flow_layers = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
        return nll_flow_layers

    flows = list(reversed(self.flows))
    flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
    z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
    for flow in flows:
        z = torch.flip(z, [1])
        z = flow(z, x_mask, g=x, reverse=reverse)

    z0, _ = torch.split(z, [1, 1], 1)
    logw = z0
    return logw
  
class StochasticEnergyPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, emoin_channels=0):
    super().__init__()
    
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.emoin_channels = emoin_channels

    # condition encoder text
    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.convs = modules.DilatedDepthSeparableConv(filter_channels, kernel_size, num_layers=3, dropout_p=p_dropout)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

    # posterior encoder
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    self.flows += [modules.ConvFlow(2, filter_channels, kernel_size, num_layers=3) for _ in range(n_flows)]

    if emoin_channels != 0:
      self.cond_emo = nn.Conv1d(emoin_channels, filter_channels, 1)

  def forward(self, x, x_mask, dr=None, g=None, emo=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)

    if emo is not None:
        emo = torch.detach(emo)
        x = x + self.cond_emo(emo)

    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
        flows = self.flows
        assert dr is not None

        noise = torch.randn(dr.size()).to(device=x.device, dtype=x.dtype) * x_mask
        z = torch.cat([dr, noise], 1)

        logdet_tot = 0
        # flow layers
        for idx, flow in enumerate(flows):
            z, logdet = flow(z, x_mask, g=x, reverse=reverse)
            logdet_tot = logdet_tot + logdet
            if idx > 0:
                z = torch.flip(z, [1])

        # flow layers - neg log likelihood
        nll_flow_layers = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
        return nll_flow_layers

    flows = list(reversed(self.flows))
    flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
    z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
    for flow in flows:
        z = torch.flip(z, [1])
        z = flow(z, x_mask, g=x, reverse=reverse)

    z0, _ = torch.split(z, [1, 1], 1)
    logw = z0
    return logw

class ProsodyDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        emoin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emoin_channels = emoin_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions_so.FFT(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(emoin_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, emo=None):
        x = torch.detach(x)
        if emo is not None:
            emo = torch.detach(emo)
            x = x + self.cond(emo)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x

class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, n_predictions=1, gin_channels=0, emoin_channels=0):
        super(TemporalPredictor, self).__init__()

        self.gin_channels = gin_channels
        self.layers = nn.Sequential(*[
            modules.ConvReLUNormFP(input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, input_size, 1)

        if emoin_channels != 0:
            self.cond_emo = nn.Conv1d(emoin_channels, input_size, 1)

    def forward(self, enc_out, enc_out_mask, g=None, emo=None):
        enc_out = torch.detach(enc_out)
        if g is not None:
            g = torch.detach(g)
            enc_out = enc_out + self.cond(g)

        if emo is not None:
            emo = torch.detach(emo)
            enc_out = enc_out + self.cond_emo(emo)

        out = enc_out * enc_out_mask
        out = self.layers(out).transpose(1, 2)
        out = self.fc(out) 
        out = out * enc_out_mask.transpose(1, 2)
        return out.squeeze(-1)
    
class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0, lin_channels=0, emoin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = attentions.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = attentions.LayerNorm(filter_channels)

    # Output Layer
    self.proj = nn.Conv1d(filter_channels, 1, 1)
    if gin_channels != 0:
        self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    if lin_channels != 0:
        self.cond_lang = nn.Conv1d(lin_channels, in_channels, 1)

    if emoin_channels != 0:
        self.cond_emo = nn.Conv1d(emoin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None, l=None, emo=None):
    x = torch.detach(x)

    if g is not None:
        g = torch.detach(g)
        x = x + self.cond(g)

    if emo is not None:
        emo = torch.detach(emo)
        x = x + self.cond_emo(emo)

    if l is not None:
        l = torch.detach(l)
        x = x + self.cond_lang(l)

    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class TextEncoder(nn.Module):
  def __init__(self, 
      n_vocab, 
      out_channels, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      n_heads, 
      n_layers, 
      kernel_size, 
      p_dropout, 
      window_size=None,
      block_length=None,
      mean_only=False,
      prenet=False,
      use_sdp=False,
      gin_channels=0,
      lin_channels=0,
      emoin_channels=0):

    super().__init__()

    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.prenet = prenet
    self.use_sdp = use_sdp
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels
    self.emoin_channels = emoin_channels

    if lin_channels > 0:
        hidden_channels -= lin_channels

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    if emoin_channels != 0:
      self.cond_emo = nn.Linear(emoin_channels, hidden_channels)

    if lin_channels > 0:
        hidden_channels += lin_channels

    if use_sdp:
      print("Use StochasticDurationPredictor")
      self.proj_w = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels, lin_channels=lin_channels, emoin_channels=emoin_channels)
    else:
      print("Use DurationPredictor")
      self.proj_w = DurationPredictor(hidden_channels, filter_channels_dp, kernel_size, p_dropout, gin_channels=gin_channels, lin_channels=lin_channels, emoin_channels=emoin_channels)

    if prenet:
      self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
    
    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      window_size=window_size,
      block_length=block_length,
      gin_channels=gin_channels,
      emoin_channels=emoin_channels
    )

    self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
    if not mean_only:
      self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_lengths, l=None, g=None, emo=None):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]

    if emo is not None:
      x = x + self.cond_emo(emo.transpose(2, 1)) # [b, 1, h]

    if l is not None:
      x = torch.cat((x, l.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if self.prenet:
      x = self.pre(x, x_mask)

    x = self.encoder(x, x_mask, g=g, emo=emo)

    x_m = self.proj_m(x) * x_mask # Stats
    if not self.mean_only:
      x_logs = self.proj_s(x) * x_mask
    else:
      x_logs = torch.zeros_like(x_m)

    #logw = self.proj_w(x_dp, x_mask)
    return x, x_m, x_logs, x_mask


class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0,
      emoin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels
    self.emoin_channels = emoin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        attentions.CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size, 
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          emoin_channels=emoin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale,
          n_sqz=n_sqz))

  def forward(self, x, x_mask, g=None, emo=None, pitch=None, energy=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)

    for f in flows:
      if not reverse:
        x, logdet = f(x, x_mask, g=g, emo=emo, pitch=pitch, energy=energy, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, emo=emo, pitch=pitch, energy=energy, reverse=reverse)

    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()


class FlowGenerator(nn.Module):
  def __init__(self, 
      n_vocab, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      out_channels,
      kernel_size=3, 
      n_heads=2, 
      n_layers_enc=10,
      p_dropout=0., 
      n_blocks_dec=12, 
      kernel_size_dec=5, 
      dilation_rate=5, 
      n_block_layers=4,
      p_dropout_dec=0., 
      n_speakers=0, 
      n_lang=0, 
      gin_channels=0, 
      lin_channels=0,
      emoin_channels=0,
      n_split=4,
      n_sqz=1,
      sigmoid_scale=False,
      window_size=None,
      block_length=None,
      mean_only=False,
      hidden_channels_enc=None,
      hidden_channels_dec=None,
      prenet=False,
      use_spk_embeds=False,
      use_lang_embeds=False,
      use_emo_embeds=False,
      use_sdp=True,
      use_spp=False,
      use_sep=False,
      ref_enc_filters=[32, 32, 64, 64, 128, 128],
      ref_enc_gru_size=128,
      token_embedding_size=256,
      token_num=10,
      num_heads=8,
      **kwargs):
    
    super().__init__()
    self.n_vocab = n_vocab
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_heads = n_heads
    self.n_layers_enc = n_layers_enc
    self.p_dropout = p_dropout
    self.n_blocks_dec = n_blocks_dec
    self.kernel_size_dec = kernel_size_dec
    self.dilation_rate = dilation_rate
    self.n_block_layers = n_block_layers
    self.p_dropout_dec = p_dropout_dec
    self.n_speakers = n_speakers
    self.n_lang = n_lang
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels
    self.emoin_channels = emoin_channels
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.hidden_channels_enc = hidden_channels_enc
    self.hidden_channels_dec = hidden_channels_dec
    self.prenet = prenet
    self.use_spk_embeds = use_spk_embeds
    self.use_lang_embeds = use_lang_embeds
    self.use_emo_embeds = use_emo_embeds
    self.use_sdp = use_sdp
    self.use_spp = use_spp
    self.use_sep = use_sep
    
    self.encoder = TextEncoder(
        n_vocab, 
        out_channels, 
        hidden_channels_enc or hidden_channels, 
        filter_channels, 
        filter_channels_dp, 
        n_heads, 
        n_layers_enc, 
        kernel_size, 
        p_dropout, 
        window_size=window_size,
        block_length=block_length,
        mean_only=mean_only,
        prenet=prenet,
        use_sdp=use_sdp,
        gin_channels=gin_channels,
        lin_channels=lin_channels, # Multi Lang
        emoin_channels=emoin_channels) # Multi Lang

    self.decoder = FlowSpecDecoder(
        out_channels, 
        hidden_channels_dec or hidden_channels, 
        kernel_size_dec, 
        dilation_rate, 
        n_blocks_dec, 
        n_block_layers, 
        p_dropout=p_dropout_dec, 
        n_split=n_split,
        n_sqz=n_sqz,
        sigmoid_scale=sigmoid_scale,
        gin_channels=gin_channels, # Multi Lang
        emoin_channels=emoin_channels) # Multi Lang

    # if self.use_spk_embeds:
    #   print("Use Speaker Embed Linear Norm")
    #   self.emb_g = nn.Linear(512, gin_channels)
    # else:
    #   if n_speakers > 1:
    #     print("Use Speaker Cathegorical")
    #     self.emb_g = nn.Embedding(n_speakers, gin_channels)
    #     nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)
    
    if self.use_lang_embeds:
      print("Use Multilanguage Cathegorical")
      self.emb_l = nn.Embedding(n_lang, lin_channels)
      torch.nn.init.xavier_uniform_(self.emb_l.weight)

    if self.use_emo_embeds:
      print("Use Raw Emo")
      #print("Use GST Custom Module")
      # self.gst_proj = GST(token_num, token_embedding_size, num_heads, ref_enc_filters, 80, ref_enc_gru_size)
      #self.gst_proj = GSTNoReff(token_num, token_embedding_size, num_heads, ref_enc_filters, 80, ref_enc_gru_size, emoin_channels=1024)

    #   print("Use Emo Catcher Custom Module")
    #   self.emo_proj = EmoCatcher(input_dim=80, hidden_dim=512, kernel_size=3, num_classes=5)
    #   self.emo_proj.load_state_dict(torch.load("/run/media/fourier/Data2/Pras/Thesis/TryModel/glow-tts/best_model_0.9170_0.5059.pth"))
    #   self.emo_proj.eval()
    #   for param in self.emo_proj.parameters():
    #     param.requires_grad = False

      #self.emo_ref = MelStyleEncoder()

    #   print("Use Emotion Embeddings Module")
    #   self.emb_emo = nn.Linear(1024, emoin_channels)

    if use_spp:
      print("Use StochasticPitchPredictor")
      # self.proj_pitch = TemporalPredictor(
      #       hidden_channels_enc,
      #       filter_size=256,
      #       kernel_size=3,
      #       dropout=0.1, 
      #       n_layers=2,
      #       n_predictions=1,
      #       gin_channels=gin_channels, emoin_channels=emoin_channels
      # )
      self.proj_pitch = StochasticPitchPredictor(hidden_channels_enc, 256, 3, 0.1, 4, emoin_channels=emoin_channels)
    else:
      print("Use Prosody Pred Pitch Updated") 
      self.proj_pitch = ProsodyDecoder(1, hidden_channels_enc, 256, 2, 6, 3, 0.1, emoin_channels=emoin_channels)

    if use_sep:
      print("Use StochasticEnergyPredictor Updated") 
      # self.proj_energy = TemporalPredictor(
      #       hidden_channels_enc,
      #       filter_size=256,
      #       kernel_size=3,
      #       dropout=0.1, 
      #       n_layers=2,
      #       n_predictions=1,
      #       emoin_channels=emoin_channels
      # )
      self.proj_energy = StochasticEnergyPredictor(hidden_channels_enc, 256, 3, 0.1, 4, emoin_channels=emoin_channels)
    else:
      print("Use Prosody Pred Energy Updated") 
      self.proj_energy = ProsodyDecoder(hidden_channels_enc, 256, 3, 0.1, 4, emoin_channels=emoin_channels)

    # for param in self.proj_pitch.parameters():
    #     param.requires_grad = False

    # for param in self.proj_energy.parameters():
    #     param.requires_grad = False

    # for param in self.encoder.parameters():
    #     param.requires_grad = False

    # for param in self.decoder.parameters():
    #     param.requires_grad = False

    # for param in self.emb_g.parameters():
    #     param.requires_grad = False

    # for param in self.emb_l.parameters():
    #     param.requires_grad = False

    # for param in self.style_encoder.parameters():
    #     param.requires_grad = False

  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, emo=None, pitch=None, energy=None, l=None):
    if g is not None:
      g = F.normalize(g).unsqueeze(-1) # [b, h, 1]

    if l is not None:
      l = self.emb_l(l).unsqueeze(-1) # [b, h, 1]

    if emo is not None:
      emo = emo.unsqueeze(-1) #self.gst_proj(emo) # [b, h, 1]

    #emo, kl_loss_emo = self.gst_proj(y)
    #emo, kl_loss_emo = self.gst_proj(y)
    # _, emo = self.emo_proj(y.unsqueeze(1), y_lengths.cpu())
    # emo = emo.unsqueeze(-1)
    #emo = self.emo_ref(y)
    
    x, x_m, x_logs, x_mask = self.encoder(x, x_lengths, l=l, g=g, emo=emo)

    y_max_length = y.size(2)
    y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    if self.use_spp and pitch is not None:
      pitch = pitch.squeeze(1)
      pitch = pitch[:, :y_max_length]
      pitch_mask = (pitch == 0.0)
      #pitch_norm = torch.log(torch.clamp(pitch, min=torch.finfo(pitch.dtype).tiny))
      pitch_norm = pitch
      pitch_norm[pitch_mask] = 0.0
      pitch_norm = pitch_norm.unsqueeze(1)

    if self.use_sep and energy is not None:
      energy = energy.squeeze(1)
      energy = energy[:, :y_max_length]
      energy_mask = (energy == 0.0)
      #energy_norm = torch.log(torch.clamp(energy, min=torch.finfo(energy.dtype).tiny))
      energy_norm = energy
      energy_norm[energy_mask] = 0.0
      energy_norm = energy_norm.unsqueeze(1)

    z, logdet = self.decoder(y, z_mask, g=g, emo=emo, pitch=pitch_norm, energy=energy_norm, reverse=False)
    with torch.no_grad():
      x_s_sq_r = torch.exp(-2 * x_logs)
      logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
      logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
      logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
      logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
      logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']
      attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.squeeze(1).sum(2).unsqueeze(1) # Attention Duration
    if self.use_sdp:
      l_length = self.encoder.proj_w(x, x_mask, w, g=g, l=l, emo=emo)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-8) * x_mask
      logw = self.encoder.proj_w(x, x_mask, g=g, l=l, emo=emo)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 

    x_feature = torch.matmul(x, attn.squeeze(1))
    if self.use_spp and pitch is not None:
      pitch_norm = pitch_norm.squeeze(1)
      l_pitch = self.proj_pitch(x_feature, z_mask, pitch_norm.unsqueeze(1), emo=emo)
      l_pitch = l_pitch / torch.sum(z_mask)
      l_pitch = torch.sum(l_pitch)
      #pred_pitch = self.proj_pitch(x_feature, z_mask, g=g, emo=emo)
    else:
      pitch_norm = pitch_norm.squeeze(1)
      pred_pitch = self.proj_pitch(x_feature, pitch_norm.unsqueeze(1), z_mask, emo=emo)
      l_pitch = F.mse_loss(pred_pitch, pitch_norm)

    if self.use_sep and energy is not None:
      energy_norm = energy_norm.squeeze(1)
      l_energy = self.proj_energy(x_feature, z_mask, energy_norm.unsqueeze(1), emo=emo)
      l_energy = l_energy / torch.sum(z_mask)
      l_energy = torch.sum(l_energy)
      #pred_energy = self.proj_energy(x_feature, z_mask, emo=emo)
    else:
      energy_norm = energy_norm.squeeze(1)
      pred_energy = self.proj_energy(x_feature, energy_norm.unsqueeze(1), z_mask, emo=emo)
      l_energy = F.mse_loss(pred_energy, energy_norm)

    # expand prior
    z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    #z_gen = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * 1.) * z_mask
    #y_gen, _ = self.decoder(z_gen, z_mask, g=g, pitch=pitch_norm, energy=energy_norm, reverse=True)

    # _, emo_gen = self.emo_proj(y_gen.unsqueeze(1), torch.tensor(y_gen.size(2)).repeat(y_gen.shape[0])) # [b, h]
    # #l_emo = torch.sum(cosine_sim(emo.squeeze(-1), emo_gen)) / emo_gen.shape[0] # [b]
    # l_emo = -torch.nn.functional.cosine_similarity(emo.squeeze(-1), emo_gen).mean()
    # l_emo = l_emo * 9.0
    # y_slice, slice_ids = commons.rand_segments(y, y_lengths, 64, let_short_samples=True, pad_short=True)
    # y_gen = commons.segment(y_gen, slice_ids, 64, pad_short=True)

    #l_pitch = None
    #l_energy = None
    return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, l_length, l_pitch, l_energy), (None, None, None, None), (None) # (attn, l_length, l_pitch, l_energy)

  def infer(self, x, x_lengths, y=None, y_lengths=None, g=None, emo=None, l=None, gst_token=None, noise_scale=1., noise_scale_w=1., f0_noise_scale=1., energy_noise_scale=1., length_scale=1., pitch_scale=1.0, energy_scale=1.0):
    if g is not None:
      g = F.normalize(g).unsqueeze(-1) # [b, h]

    if l is not None:
      l = self.emb_l(l).unsqueeze(-1) # [b, h]

    if emo is not None:
      emo = emo.unsqueeze(-1) #self.gst_proj(emo) # [b, h, 1]
    
    # if gst_token is not None:
    #    emo = gst_token
    # elif y is not None:
    #   emo, _ = self.gst_proj(y, infer=True)
    #emo = self.emo_ref(y)
    # _, emo = self.emo_proj(y.unsqueeze(1), y_lengths.cpu())
    # emo = emo.unsqueeze(-1)
    
    x, x_m, x_logs, x_mask = self.encoder(x, x_lengths, l=l, g=g, emo=emo)

    if self.use_sdp:
      logw = self.encoder.proj_w(x, x_mask, g=g, l=l, emo=emo, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.encoder.proj_w(x, x_mask, g=g, l=l, emo=emo)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    #y_max_length = None

    #y, y_lengths, y_max_length = self.preprocess(None, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

    z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
    
    x_feature = torch.matmul(x, attn.squeeze(1))
    if self.use_spp:
      pitch = self.proj_pitch(x_feature, z_mask, emo=emo, noise_scale=f0_noise_scale, reverse=True)
      #pitch = self.proj_pitch(x_feature, z_mask, g=g, emo=emo)
      pitch = pitch.squeeze(1)
      #pitch = torch.clamp_min(pitch, 0)
      if pitch.shape[-1] != z.shape[-1]:
        # need to expand predicted pitch to match no of tokens
        durs_predicted = torch.sum(attn, -1) * x_mask.squeeze()
        pitch, _ = commons.regulate_len(durs_predicted, pitch.unsqueeze(-1))
        pitch = pitch.squeeze(-1) 
      pitch = pitch * pitch_scale
      pitch = pitch.squeeze(1)

    if self.use_sep:
      energy = self.proj_energy(x_feature, z_mask, emo=emo, noise_scale=energy_noise_scale, reverse=True)
      #energy = self.proj_energy(x_feature, z_mask, emo=emo)
      energy = energy.squeeze(1)
      #energy = torch.clamp_min(energy, 0)
      if energy.shape[-1] != z.shape[-1]:
        # need to expand predicted energy to match no of tokens
        durs_predicted = torch.sum(attn, -1) * x_mask.squeeze()
        energy, _ = commons.regulate_len(durs_predicted, energy.unsqueeze(-1))
        energy = energy.squeeze(-1)    
      energy = energy * energy_scale
      energy = energy.squeeze(1) 
    
    y, logdet = self.decoder(z, z_mask, g=g, emo=emo, pitch=pitch, energy=energy, reverse=True)
    return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_), (pitch, energy)

  def voice_conversion(self, y, y_lengths, spk_embed_src, spk_embed_tgt, l=None):
    g_src = self.emb_g(spk_embed_src).unsqueeze(-1)
    g_tgt = self.emb_g(spk_embed_tgt).unsqueeze(-1)

    if l is not None:
      l = F.normalize(self.emb_l(l)).unsqueeze(-1) # [b, h]
      g_src = torch.cat([g_src, l], 1)
      g_tgt = torch.cat([g_tgt, l], 1)

    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1)
    z, _ = self.decoder(y, z_mask, g=g_src, reverse=False)

    y_conv, _ = self.decoder(z, z_mask, g=g_tgt, reverse=True)
    return y_conv
  
  def preprocess(self, y, y_lengths, y_max_length):
    if y_max_length is not None:
      y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
      y = y[:,:,:y_max_length]
    y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
    return y, y_lengths, y_max_length

  def store_inverse(self):
    self.decoder.store_inverse()

def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg
