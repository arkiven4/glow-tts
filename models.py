import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons
import attentions
import monotonic_align

class CVAE_Emo(nn.Module):
    def __init__(self, feature_size, latent_size, hidden_state=96, class_size=3):
        super(CVAE_Emo, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_state = hidden_state
        self.class_size = class_size

        self.fc1_a  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_a = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_a = modules.LinearNorm(hidden_state, latent_size)

        self.fc1_d  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_d = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_d = modules.LinearNorm(hidden_state, latent_size)

        self.fc1_v  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_v = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_v = modules.LinearNorm(hidden_state, latent_size)

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu_a, logvar_a = self.encode_a(x[:,0] - 1) # -1 because when precessing accidentaly adding 1
        mu_d, logvar_d = self.encode_d(x[:,0] - 1)
        mu_v, logvar_v = self.encode_v(x[:,0] - 1)

        z_a = self.reparameterize(mu_a, logvar_a)
        z_d = self.reparameterize(mu_d, logvar_d)
        z_v = self.reparameterize(mu_v, logvar_v)

        return z_a + z_d + z_v

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0, lin_channels=0):
    super().__init__()
    if lin_channels != 0 :
      in_channels += lin_channels
    
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    if lin_channels != 0 :
      self.cond_lang = nn.Conv1d(lin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, l=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)

    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    if l is not None:
      l = torch.detach(l)
      x = x + self.cond_lang(l)

    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b] # stoch_dur_loss
    
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw # log_durs_predicted
    
class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0, lin_channels=0):
    super().__init__()
    if lin_channels != 0 :
      in_channels += lin_channels

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    if lin_channels != 0 :
      self.cond_lang = nn.Conv1d(lin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None, l=None):
    x = torch.detach(x)

    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

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

# class DurationPredictor(nn.Module):
#   def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
#     super().__init__()

#     self.in_channels = in_channels
#     self.filter_channels = filter_channels
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout

#     self.drop = nn.Dropout(p_dropout)
#     self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
#     self.norm_1 = attentions.LayerNorm(filter_channels)
#     self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
#     self.norm_2 = attentions.LayerNorm(filter_channels)
#     self.proj = nn.Conv1d(filter_channels, 1, 1)

#   def forward(self, x, x_mask):
#     x = self.conv_1(x * x_mask)
#     x = torch.relu(x)
#     x = self.norm_1(x)
#     x = self.drop(x)
#     x = self.conv_2(x * x_mask)
#     x = torch.relu(x)
#     x = self.norm_2(x)
#     x = self.drop(x)
#     x = self.proj(x * x_mask)
#     return x * x_mask


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
      lin_channels=0):

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
    self.lin_channels = lin_channels

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    if lin_channels > 0:
        hidden_channels += lin_channels

    #self.emo_proj = modules.LinearNorm(1024, hidden_channels) #Follow emoin_channels
    self.emo_proj_a = modules.LinearNorm(1, hidden_channels)
    self.emo_proj_d = modules.LinearNorm(1, hidden_channels)
    self.emo_proj_v = modules.LinearNorm(1, hidden_channels)

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
    )

    self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
    if not mean_only:
      self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_lengths, l=None, emo=None):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]

    if l is not None:
      x = torch.cat((x, l.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
    
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if self.prenet:
      x = self.pre(x, x_mask)
    x = self.encoder(x, x_mask)
    
    if emo is not None:
      emb_a = self.emo_proj_a(emo[:,0].unsqueeze(1))
      emb_d = self.emo_proj_d(emo[:,1].unsqueeze(1))
      emb_v = self.emo_proj_v(emo[:,2].unsqueeze(1))

      emb_emo = emb_a + emb_d + emb_v
      x = x + emb_emo.unsqueeze(2) # [b, t, h]

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
      gin_channels=0):
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
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale))

  def forward(self, x, x_mask, g=None, reverse=False):
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
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
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
      n_layers_enc=6,
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
        lin_channels=lin_channels) # Multi Lang

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
        gin_channels=gin_channels) # Multi Lang

    if self.use_spk_embeds:
      print("Use Speaker Embed Linear Norm")
      self.emb_g = modules.LinearNorm(512, gin_channels)
    else:
      if n_speakers > 1:
        print("Use Speaker Cathegorical")
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)
    
    if self.use_lang_embeds:
      print("Use Multilanguage Cathegorical")
      self.emb_l = nn.Embedding(n_lang, lin_channels)
      torch.nn.init.xavier_uniform_(self.emb_l.weight)
      #nn.init.uniform_(self.emb_l.weight, -0.1, 0.1)

    # if self.use_emo_embeds:
    #   print("Use Emotion Embedding")
      #self.emb_emo = CVAE_Emo(1, hidden_channels_enc, 96)
      #self.emb_emo = modules.LinearNorm(1024, emoin_channels)

    if use_sdp:
      print("Use SDP")
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels, lin_channels=lin_channels)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels, lin_channels=lin_channels)

  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, emo=None, l=None):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]

    if l is not None:
      l = self.emb_l(l).unsqueeze(-1)

    x, x_m, x_logs, x_mask = self.encoder(x, x_lengths, l=l, emo=emo)

    y_max_length = y.size(2)

    y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
    with torch.no_grad():
      x_s_sq_r = torch.exp(-2 * x_logs)
      logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
      logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
      logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
      logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
      logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

      attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.squeeze(1).sum(2).unsqueeze(1)
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g, l=l)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.dp(x, x_mask, g=g, l=l)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 

    # expand prior
    z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    
    return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, l_length)

  def infer(self, x, x_lengths, y=None, y_lengths=None, g=None, emo=None, l=None, noise_scale=1., length_scale=1.):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]

    if l is not None:
      l = self.emb_l(l).unsqueeze(-1)

    x, x_m, x_logs, x_mask = self.encoder(x, x_lengths, l=l, emo=emo)

    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, l=l, reverse=True, noise_scale=noise_scale)
    else:
      logw = self.dp(x, x_mask, g=g, l=l)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max_length = None

    y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
    attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    
    z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

    z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
    y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
    return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)

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
