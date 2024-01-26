import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons
import attentions
import monotonic_align

class Styling_Emotion(nn.Module):
    def __init__(self, feature_size, latent_size, hidden_state=96):
        super(Styling_Emotion, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_state = hidden_state

        self.fc1_a  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_a = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_a = modules.LinearNorm(hidden_state, latent_size)

        self.fc1_d  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_d = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_d = modules.LinearNorm(hidden_state, latent_size)

        self.fc1_v  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_v = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_v = modules.LinearNorm(hidden_state, latent_size)

        self.fc1_ad  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_ad = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_ad = modules.LinearNorm(hidden_state, latent_size)

        self.fc1_vd  = modules.LinearNorm(feature_size, hidden_state)
        self.fc21_vd = modules.LinearNorm(hidden_state, latent_size)
        self.fc22_vd = modules.LinearNorm(hidden_state, latent_size)

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
    
    def encode_ad(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = x.unsqueeze(1) # (bs, coordinate)
        h1 = self.elu(self.fc1_ad(inputs))
        z_mu = self.fc21_ad(h1)
        z_var = self.fc22_ad(h1)
        return z_mu, z_var
    
    def encode_vd(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = x.unsqueeze(1) # (bs, coordinate)
        h1 = self.elu(self.fc1_vd(inputs))
        z_mu = self.fc21_vd(h1)
        z_var = self.fc22_vd(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        arousal_input = x[:,0] - 1
        valence_input = x[:,2] - 1
        dominance_input = x[:,1] - 1

        a_perd = arousal_input / dominance_input
        v_perd = valence_input / dominance_input

        mu_a, logvar_a = self.encode_a(arousal_input) # -1 because when precessing accidentaly adding 1
        mu_d, logvar_d = self.encode_d(dominance_input)
        mu_v, logvar_v = self.encode_v(valence_input)

        mu_ad, logvar_ad = self.encode_ad(a_perd)
        mu_vd, logvar_vd = self.encode_vd(v_perd)

        z_a = self.reparameterize(mu_a, logvar_a)
        z_d = self.reparameterize(mu_d, logvar_d)
        z_v = self.reparameterize(mu_v, logvar_v)
        z_ad = self.reparameterize(mu_ad, logvar_ad)
        z_vd = self.reparameterize(mu_vd, logvar_vd)

        return z_a + z_d + z_v + z_ad + z_vd

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
      self.cond_emo = nn.Conv1d(emoin_channels, in_channels, 1)

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
      use_sdp=False, 
      gin_channels=0,
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
    self.use_sdp = use_sdp
    self.gin_channels = gin_channels
    self.lin_channels = lin_channels

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    if lin_channels > 0:
        hidden_channels += lin_channels

    if use_sdp:
      print("Use StochasticDurationPredictor")
      self.proj_w = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels, lin_channels=lin_channels, emoin_channels=hidden_channels-lin_channels)
    else:
      print("Use DurationPredictor")
      self.proj_w = DurationPredictor(hidden_channels, filter_channels_dp, kernel_size, p_dropout, gin_channels=gin_channels, lin_channels=lin_channels, emoin_channels=hidden_channels-lin_channels)

    #self.emo_proj = modules.LinearNorm(1024, hidden_channels) #Follow emoin_channels
    # self.emo_proj_a = modules.LinearNorm(1, hidden_channels)
    # self.emo_proj_d = modules.LinearNorm(1, hidden_channels)
    # self.emo_proj_v = modules.LinearNorm(1, hidden_channels)

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

    if emo is not None:
      x = x + emo.transpose(2, 1) # [b, 1, h]

    if l is not None:
      x = torch.cat((x, l.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
    
    # if emo is not None:
    #   emb_a = self.emo_proj_a(emo[:,0].unsqueeze(1))
    #   emb_d = self.emo_proj_d(emo[:,1].unsqueeze(1))
    #   emb_v = self.emo_proj_v(emo[:,2].unsqueeze(1))

    #   emb_emo = emb_a + emb_d + emb_v
    #   emb_emo = emb_emo.unsqueeze(1)
    #   x = x + emb_emo # [b, t, h]

    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if self.prenet:
      x = self.pre(x, x_mask)
    x = self.encoder(x, x_mask)

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
        use_sdp=use_sdp,
        gin_channels=gin_channels,
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

    if self.use_emo_embeds:
      print("Use Emotion Embedding")
      self.emb_emo = Styling_Emotion(1, hidden_channels_enc, 96)

  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, emo=None, l=None):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]

    if l is not None:
      l = F.normalize(self.emb_l(l)).unsqueeze(-1) # [b, h, 1]
      #g = torch.cat([g, l], 1)

    if emo is not None:
      emo = F.normalize(self.emb_emo(emo)).unsqueeze(-1) # [b, h, 1]

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

    w = attn.squeeze(1).sum(2).unsqueeze(1) # Attention Duration
    if self.use_sdp:
      l_length = self.encoder.proj_w(x, x_mask, w, g=g, l=l, emo=emo)
      l_length = l_length / torch.sum(x_mask)
    else:
      # if g is not None:
      #   g_exp = g.expand(-1, -1, x.size(-1))
      #   x_dp = torch.cat([torch.detach(x), g_exp], 1)
      # else:
      #   x_dp = torch.detach(x)

      logw_ = torch.log(w + 1e-8) * x_mask
      logw = self.encoder.proj_w(x, x_mask, g=g, l=l, emo=emo)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 

    # expand prior
    z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    
    return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, l_length)

  def infer(self, x, x_lengths, g=None, emo=None, l=None, noise_scale=1., length_scale=1.):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]

    if l is not None:
      l = F.normalize(self.emb_l(l)).unsqueeze(-1) # [b, h]
      #g = torch.cat([g, l], 1)

    if emo is not None:
      emo = F.normalize(self.emb_emo(emo)).unsqueeze(-1) # [b, h, 1]

    x, x_m, x_logs, x_mask = self.encoder(x, x_lengths, l=l, emo=emo)

    if self.use_sdp:
      logw = self.encoder.proj_w(x, x_mask, g=g, l=l, emo=emo, reverse=True, noise_scale=noise_scale)
    else:
      # if g is not None:
      #   g_exp = g.expand(-1, -1, x.size(-1))
      #   x_dp = torch.cat([torch.detach(x), g_exp], 1)
      # else:
      #   x_dp = torch.detach(x)

      logw = self.encoder.proj_w(x, x_mask, g=g, l=l, emo=emo)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max_length = None

    y, y_lengths, y_max_length = self.preprocess(None, y_lengths, y_max_length)
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
