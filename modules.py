import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from transforms import piecewise_rational_quadratic_transform

import commons


class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-4):
      super().__init__()
      self.channels = channels
      self.eps = eps

      self.gamma = nn.Parameter(torch.ones(channels))
      self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    n_dims = len(x.shape)
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x -mean)**2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + self.eps)

    shape = [1, -1] + [1] * (n_dims - 2)
    x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x

class LayerNorm2(nn.Module):
    """Layer norm for the 2nd dimension of the input using torch primitive.
    Args:
        channels (int): number of channels (2nd dimension) of the input.
        eps (float): to prevent 0 division

    Shapes:
        - input: (B, C, T)
        - output: (B, C, T)
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

class ConvReluNorm(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    assert n_layers > 1, "Number of layers should be larger than 0."

    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    for _ in range(n_layers-1):
      self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x, x_mask):
    x_org = x
    for i in range(self.n_layers):
      x = self.conv_layers[i](x * x_mask)
      x = self.norm_layers[i](x)
      x = self.relu_drop(x)
    x = x_org + self.proj(x)
    return x * x_mask


class WN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
      super(WN, self).__init__()
      assert(kernel_size % 2 == 1)
      assert(hidden_channels % 2 == 0)
      self.in_channels = in_channels
      self.hidden_channels =hidden_channels
      self.kernel_size = kernel_size,
      self.dilation_rate = dilation_rate
      self.n_layers = n_layers
      self.gin_channels = gin_channels
      self.p_dropout = p_dropout

      self.in_layers = torch.nn.ModuleList()
      self.res_skip_layers = torch.nn.ModuleList()
      self.drop = nn.Dropout(p_dropout)

      if gin_channels != 0:
        cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

      for i in range(n_layers):
        dilation = dilation_rate ** i
        padding = int((kernel_size * dilation - dilation) / 2)
        in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                   dilation=dilation, padding=padding)
        in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
        self.in_layers.append(in_layer)

        # last one is not necessary
        if i < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask=None, g=None, **kwargs):
      output = torch.zeros_like(x)
      n_channels_tensor = torch.IntTensor([self.hidden_channels])

      if g is not None:
        g = self.cond_layer(g)

      for i in range(self.n_layers):
          x_in = self.in_layers[i](x)
          x_in = self.drop(x_in)
          if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            g_l = torch.zeros_like(x_in)

          acts = commons.fused_add_tanh_sigmoid_multiply(
              x_in,
              g_l,
              n_channels_tensor)

          res_skip_acts = self.res_skip_layers[i](acts)
          if i < self.n_layers - 1:
            x = (x + res_skip_acts[:,:self.hidden_channels,:]) * x_mask
            output = output + res_skip_acts[:,self.hidden_channels:,:]
          else:
            output = output + res_skip_acts
      return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)

class WNGE(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, emoin_channels=0, p_dropout=0):
      super(WNGE, self).__init__()
      assert(kernel_size % 2 == 1)
      assert(hidden_channels % 2 == 0)
      self.in_channels = in_channels
      self.hidden_channels =hidden_channels
      self.kernel_size = kernel_size,
      self.dilation_rate = dilation_rate
      self.n_layers = n_layers
      self.gin_channels = gin_channels
      self.p_dropout = p_dropout

      self.in_layers = torch.nn.ModuleList()
      self.res_skip_layers = torch.nn.ModuleList()
      self.drop = nn.Dropout(p_dropout)

      if gin_channels != 0:
        cond_layer_g = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer_g = torch.nn.utils.weight_norm(cond_layer_g, name='weight')

      if emoin_channels != 0:
        cond_layer_emo = torch.nn.Conv1d(emoin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer_emo = torch.nn.utils.weight_norm(cond_layer_emo, name='weight')

      for i in range(n_layers):
        dilation = dilation_rate ** i
        padding = int((kernel_size * dilation - dilation) / 2)
        in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                   dilation=dilation, padding=padding)
        in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
        self.in_layers.append(in_layer)

        # last one is not necessary
        if i < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask=None, g=None, emo=None, **kwargs):
      output = torch.zeros_like(x)
      n_channels_tensor = torch.IntTensor([self.hidden_channels])

      if g is not None:
        g = self.cond_layer_g(g)

      if emo is not None:
        emo = self.cond_layer_emo(emo)

      for i in range(self.n_layers):
          x_in = self.in_layers[i](x)
          x_in = self.drop(x_in)
          if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            g_l = torch.zeros_like(x_in)

          if emo is not None:
            cond_offset = i * 2 * self.hidden_channels
            emo_l = emo[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            emo_l = torch.zeros_like(x_in)

          acts = commons.fused_add_tanh_sigmoid_multiply(
              x_in,
              g_l + emo_l,
              n_channels_tensor)

          res_skip_acts = self.res_skip_layers[i](acts)
          if i < self.n_layers - 1:
            x = (x + res_skip_acts[:,:self.hidden_channels,:]) * x_mask
            output = output + res_skip_acts[:,self.hidden_channels:,:]
          else:
            output = output + res_skip_acts
      return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer_g)
    if self.emoin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer_emo)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)

class WNP(torch.nn.Module):
    def __init__(
        self, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels1=0, n_sqz=2,
    ):
        super(WNP, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.n_sqz = n_sqz
        self.gin_channels1 = gin_channels1

        if gin_channels1 != 0:
            cond_layer1 = torch.nn.Conv1d(gin_channels1, 2 * hidden_channels * n_layers // self.n_sqz, 1)
            self.cond_layer1 = torch.nn.utils.weight_norm(cond_layer1, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g1=None, g2=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g1 is not None:
            g1 = self.cond_layer1(g1)
            if self.n_sqz > 1:
                g1 = self.squeeze(g1, self.n_sqz)
        else:
            return x
        
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if g1 is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l1 = g1[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l1 = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l1, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels1 != 0:
          torch.nn.utils.remove_weight_norm(self.cond_layer1)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)
            
    def squeeze(self, x, n_sqz=2):
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // n_sqz, n_sqz)
        
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

        return x_sqz

class WN_Combine(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, emoin_channels=0, p_dropout=0):
      super(WN_Combine, self).__init__()
      assert(kernel_size % 2 == 1)
      assert(hidden_channels % 2 == 0)
      self.in_channels = in_channels
      self.hidden_channels =hidden_channels
      self.kernel_size = kernel_size,
      self.dilation_rate = dilation_rate
      self.n_layers = n_layers
      self.gin_channels = gin_channels
      self.p_dropout = p_dropout

      self.in_layers = torch.nn.ModuleList()
      self.res_skip_layers_g = torch.nn.ModuleList()
      self.res_skip_layers_emo = torch.nn.ModuleList()
      self.drop = nn.Dropout(p_dropout)

      if gin_channels != 0:
        cond_layer_g = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer_g = torch.nn.utils.weight_norm(cond_layer_g, name='weight')

      if emoin_channels != 0:
        cond_layer_emo = torch.nn.Conv1d(emoin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer_emo = torch.nn.utils.weight_norm(cond_layer_emo, name='weight')

      for i in range(n_layers):
        dilation = dilation_rate ** i
        padding = int((kernel_size * dilation - dilation) / 2)
        in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size, dilation=dilation, padding=padding)
        in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
        self.in_layers.append(in_layer)

        # last one is not necessary
        if i < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers_g.append(res_skip_layer)
        self.res_skip_layers_emo.append(res_skip_layer)

  def forward(self, x, x_mask=None, g=None, emo=None, **kwargs):
      output = torch.zeros_like(x)
      n_channels_tensor = torch.IntTensor([self.hidden_channels])

      if g is not None:
        g = self.cond_layer_g(g)

      if emo is not None:
        emo = self.cond_layer_emo(emo)

      for i in range(self.n_layers):
          x_in = self.in_layers[i](x)
          x_in = self.drop(x_in)
          if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            g_l = torch.zeros_like(x_in)

          acts_g = commons.fused_add_tanh_sigmoid_multiply(
              x_in,
              g_l,
              n_channels_tensor)
          
          if emo is not None:
            cond_offset = i * 2 * self.hidden_channels
            emo_l = emo[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            emo_l = torch.zeros_like(x_in)

          acts_emo = commons.fused_add_tanh_sigmoid_multiply(
              x_in,
              emo_l,
              n_channels_tensor)

          res_skip_acts_g = self.res_skip_layers_g[i](acts_g)
          res_skip_acts_emo = self.res_skip_layers_emo[i](acts_emo)
          if i < self.n_layers - 1:
            x = (x + res_skip_acts_g[:,:self.hidden_channels,:] + res_skip_acts_emo[:,:self.hidden_channels,:]) * x_mask
            output = output + res_skip_acts_g[:,self.hidden_channels:,:] + res_skip_acts_emo[:,self.hidden_channels:,:]
          else:
            output = output + res_skip_acts_g + res_skip_acts_emo
      return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer_g)
    if self.emoin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer_emo)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
      torch.nn.utils.remove_weight_norm(l)

class ActNorm(nn.Module):
  def __init__(self, channels, ddi=False, **kwargs):
    super().__init__()
    self.channels = channels
    self.initialized = not ddi

    self.logs = nn.Parameter(torch.zeros(1, channels, 1))
    self.bias = nn.Parameter(torch.zeros(1, channels, 1))

  def forward(self, x, x_mask=None, reverse=False, **kwargs):
    if x_mask is None:
      x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
    x_len = torch.sum(x_mask, [1, 2])
    if not self.initialized:
      self.initialize(x, x_mask)
      self.initialized = True

    if reverse:
      z = (x - self.bias) * torch.exp(-self.logs) * x_mask
      logdet = None
    else:
      z = (self.bias + torch.exp(self.logs) * x) * x_mask
      logdet = torch.sum(self.logs) * x_len # [b]

    return z, logdet

  def store_inverse(self):
    pass

  def set_ddi(self, ddi):
    self.initialized = not ddi

  def initialize(self, x, x_mask):
    with torch.no_grad():
      denom = torch.sum(x_mask, [0, 2])
      m = torch.sum(x * x_mask, [0, 2]) / denom
      m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
      v = m_sq - (m ** 2)
      logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

      bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
      logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

      self.bias.data.copy_(bias_init)
      self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
  def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
    super().__init__()
    assert(n_split % 2 == 0)
    self.channels = channels
    self.n_split = n_split
    self.no_jacobian = no_jacobian
    
    w_init = torch.linalg.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
    if torch.det(w_init) < 0:
      w_init[:,0] = -1 * w_init[:,0]
    self.weight = nn.Parameter(w_init)

  def forward(self, x, x_mask=None, reverse=False, **kwargs):
    b, c, t = x.size()
    assert(c % self.n_split == 0)
    if x_mask is None:
      x_mask = 1
      x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
    else:
      x_len = torch.sum(x_mask, [1, 2])

    x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

    if reverse:
      if hasattr(self, "weight_inv"):
        weight = self.weight_inv
      else:
        weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
      logdet = None
    else:
      weight = self.weight
      if self.no_jacobian:
        logdet = 0
      else:
        logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len # [b]

    weight = weight.view(self.n_split, self.n_split, 1, 1)
    z = F.conv2d(x, weight)

    z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
    z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
    return z, logdet

  def store_inverse(self):
    self.weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
    
class DilatedDepthSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, dropout_p=0.0) -> torch.tensor:
        """Dilated Depth-wise Separable Convolution module.

        ::
            x |-> DDSConv(x) -> LayerNorm(x) -> GeLU(x) -> Conv1x1(x) -> LayerNorm(x) -> GeLU(x) -> + -> o
              |-------------------------------------------------------------------------------------^

        Args:
            channels ([type]): [description]
            kernel_size ([type]): [description]
            num_layers ([type]): [description]
            dropout_p (float, optional): [description]. Defaults to 0.0.

        Returns:
            torch.tensor: Network output masked by the input sequence mask.
        """
        super().__init__()
        self.num_layers = num_layers

        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm2(channels))
            self.norms_2.append(LayerNorm2(channels))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        if g is not None:
            x = x + g
        for i in range(self.num_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.dropout(y)
            x = x + y
        return x * x_mask


class ElementwiseAffine(nn.Module):
    """Element-wise affine transform like no-population stats BatchNorm alternative.

    Args:
        channels (int): Number of input tensor channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):  # pylint: disable=unused-argument
        if not reverse:
            y = (x * torch.exp(self.log_scale) + self.translation) * x_mask
            logdet = torch.sum(self.log_scale * x_mask, [1, 2])
            return y, logdet
        x = (x - self.translation) * torch.exp(-self.log_scale) * x_mask
        return x


class ConvFlow(nn.Module):
    """Dilated depth separable convolutional based spline flow.

    Args:
        in_channels (int): Number of input tensor channels.
        hidden_channels (int): Number of in network channels.
        kernel_size (int): Convolutional kernel size.
        num_layers (int): Number of convolutional layers.
        num_bins (int, optional): Number of spline bins. Defaults to 10.
        tail_bound (float, optional): Tail bound for PRQT. Defaults to 5.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.hidden_channels = hidden_channels
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.convs = DilatedDepthSeparableConv(hidden_channels, kernel_size, num_layers, dropout_p=0.0)
        self.proj = nn.Conv1d(hidden_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.hidden_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.hidden_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        return x
    
class Mish(nn.Module):
  def __init__(self):
      super(Mish, self).__init__()
  def forward(self, x):
      return x * torch.tanh(F.softplus(x))
    
class Conv1dGLU(nn.Module):
  '''
  Conv1d + GLU(Gated Linear Unit) with residual connection.
  For GLU refer to https://arxiv.org/abs/1612.08083 paper.
  '''
  def __init__(self, in_channels, out_channels, kernel_size, dropout):
      super(Conv1dGLU, self).__init__()
      self.out_channels = out_channels
      self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
      self.dropout = nn.Dropout(dropout)
          
  def forward(self, x):
      residual = x
      x = self.conv1(x)
      x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
      x = x1 * torch.sigmoid(x2)
      x = residual + self.dropout(x)
      return x

class ConvNorm(nn.Module):
  def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=None,
                dilation=1,
                bias=True, 
                spectral_norm=False,
                ):
      super(ConvNorm, self).__init__()

      if padding is None:
          assert(kernel_size % 2 == 1)
          padding = int(dilation * (kernel_size - 1) / 2)

      self.conv = torch.nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)
      
      if spectral_norm:
          self.conv = nn.utils.spectral_norm(self.conv)

  def forward(self, input):
      out = self.conv(input)
      return out

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn
    
class ConvReLUNormFP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNormFP, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)