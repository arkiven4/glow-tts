import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from modules import LayerNorm

class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=None, block_length=None, gin_channels=0, emoin_channels=0, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout, block_length=block_length))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

    if gin_channels != 0:
      self.cond_g = nn.Linear(gin_channels, hidden_channels)

    # if emoin_channels != 0:
    #   self.cond_emo = nn.Linear(emoin_channels, hidden_channels)

    # if lin_channels != 0:
    #   self.cond_l = modules.LinearNorm(lin_channels, hidden_channels)
      
    # if gin_channels != 0:
    #     cond_layer_g = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
    #     self.cond_layer_g = torch.nn.utils.weight_norm(cond_layer_g, name='weight')
    #     self.cond_pre_g = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)

    if emoin_channels != 0:
        print("Using Emotion in Encoder...")
        cond_layer_emo = torch.nn.Conv1d(emoin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer_emo = torch.nn.utils.weight_norm(cond_layer_emo, name='weight')
        self.cond_pre_emo = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)

  def forward(self, x, x_mask, g=None, emo=None):
    # if g is not None:
    #   g = self.cond_layer_g(g)

    if emo is not None:
      emo = self.cond_layer_emo(emo)
  
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    for i in range(self.n_layers):
      x = x * x_mask
      if i == 3 - 1 and g is not None:
        x = x + self.cond_g(g.transpose(2, 1)).transpose(2, 1)

      # if i == 4 - 1 and emo is not None:
      #   x = x + self.cond_emo(emo.transpose(2, 1)).transpose(2, 1)

      if emo is not None:
        x = self.cond_pre_emo(x)
        cond_offset = i * 2 * self.hidden_channels
        emo_l = emo[:,cond_offset:cond_offset+2*self.hidden_channels,:]
        x = commons.fused_add_tanh_sigmoid_multiply(x, emo_l, torch.IntTensor([self.hidden_channels]))

      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


class CouplingBlock(nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, emoin_channels=0, p_dropout=0, sigmoid_scale=False, n_sqz=2):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.emoin_channels = emoin_channels
    self.p_dropout = p_dropout
    self.sigmoid_scale = sigmoid_scale

    start = torch.nn.Conv1d(in_channels//2, hidden_channels, 1)
    start = torch.nn.utils.weight_norm(start)
    self.start = start
    # Initializing last layer to 0 makes the affine coupling layers
    # do nothing at first.  It helps to stabilze training.
    end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
    end.weight.data.zero_()
    end.bias.data.zero_()
    self.end = end

    #self.wn = modules.WNGE(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, emoin_channels, p_dropout=p_dropout)
    #self.wn = modules.WN_Combine(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, emoin_channels, p_dropo
    self.wn_pitch = modules.WNP(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout, 1, n_sqz)
    self.wn_energy = modules.WNP(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout, 1, n_sqz)
    #self.wn_prosody = modules.WNProsody(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout, 1, 1, n_sqz)
    self.wn = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)
    #self.wn_emo = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, emoin_channels, p_dropout)
    #self.wn_prosody = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, hidden_channels, p_dropout)
    #self.wn_emo = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, emoin_channels, p_dropout)

    # self.pre_transformer = Encoder(
    #             in_channels//2,
    #             in_channels//2,
    #             n_heads=2,
    #             n_layers=1,
    #             kernel_size=3,
    #             dropout=0.1,
    #             window_size=None,
    #         )

  def forward(self, x, x_mask=None, reverse=False, g=None, emo=None, pitch=None, energy=None, **kwargs):
    b, c, t = x.size()
    if x_mask is None:
      x_mask = 1

    if pitch is not None and len(pitch.shape) == 2:
      pitch = pitch.unsqueeze(1) # B, T -> B,C,T

    if energy is not None and len(energy.shape) == 2:
      energy = energy.unsqueeze(1) # B, T -> B,C,T

    
    x_0, x_1 = x[:,:self.in_channels//2], x[:,self.in_channels//2:]

    # x_0_ = x_0
    # if self.pre_transformer is not None:
    #     x_0_ = self.pre_transformer(x_0 * x_mask, x_mask)
    #     x_0_ = x_0_ + x_0  # residual connection
    x = self.start(x_0) * x_mask
    
    #x = self.wn(x, x_mask, g, emo)
    x = self.wn(x, x_mask, g) 
    #x = self.wn_emo(x, x_mask, emo) 
    #x = self.wn_prosody(x, x_mask, pitch, energy) 
    #x = self.wn_emo(x, x_mask, emo) sss
    x = self.wn_energy(x, x_mask, energy) 
    x = self.wn_pitch(x, x_mask, pitch)
    #x = self.wn_emo(x, x_mask, emo)
    # x = self.wn_energy(x, x_mask, energy)
    # x = self.wn_pitch(x, x_mask, pitch)
    # if emo is not None:
    #   x = self.wn_emo(x, x_mask, emo)
    # if energy is not None:
    #   x = self.wn_energy(x, x_mask, energy)
    # if pitch is not None:
    #   x = self.wn_pitch(x, x_mask, pitch)
    out = self.end(x)

    z_0 = x_0
    m = out[:, :self.in_channels//2, :]
    logs = out[:, self.in_channels//2:, :]
    if self.sigmoid_scale:
      logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

    if reverse:
      z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
      logdet = None
    else:
      z_1 = (m + torch.exp(logs) * x_1) * x_mask
      logdet = torch.sum(logs * x_mask, [1, 2])

    z = torch.cat([z_0, z_1], 1)
    return z, logdet

  def store_inverse(self):
    self.wn.remove_weight_norm()
    #self.wn_emo.remove_weight_norm()
    #self.wn_prosody.remove_weight_norm()
    #self.wn_emo.remove_weight_norm()
    self.wn_energy.remove_weight_norm()
    self.wn_pitch.remove_weight_norm()


class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0., block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.p_dropout = p_dropout
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    if proximal_init:
      self.conv_k.weight.data.copy_(self.conv_q.weight.data)
      self.conv_k.bias.data.copy_(self.conv_q.bias.data)
    nn.init.xavier_uniform_(self.conv_v.weight)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x
    
  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
      rel_logits = self._relative_position_to_absolute_position(rel_logits)
      scores_local = rel_logits / math.sqrt(self.k_channels)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores * block_mask + -1e4*(1 - block_mask)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    return x * x_mask
  
