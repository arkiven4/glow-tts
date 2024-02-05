import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT


def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def mle_loss(z, m, logs, logdet, mask):
  l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
  l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l


def duration_loss(logw, logw_, lengths):
  l = torch.sum((logw - logw_)**2) / torch.sum(lengths)
  return l

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

def sus_loss(z_q):
  sus_loss = (torch.sqrt(torch.sum((z_q)**2)) - 1)**2
  return sus_loss

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x


def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def maximum_path(value, mask, max_neg_val=-np.inf):
  """ Numpy-friendly version. It's about 4 times faster than torch version.
  value: [b, t_x, t_y]
  mask: [b, t_x, t_y]
  """
  value = value * mask

  device = value.device
  dtype = value.dtype
  value = value.cpu().detach().numpy()
  mask = mask.cpu().detach().numpy().astype(np.bool)
  
  b, t_x, t_y = value.shape
  direction = np.zeros(value.shape, dtype=np.int64)
  v = np.zeros((b, t_x), dtype=np.float32)
  x_range = np.arange(t_x, dtype=np.float32).reshape(1,-1)
  for j in range(t_y):
    v0 = np.pad(v, [[0,0],[1,0]], mode="constant", constant_values=max_neg_val)[:, :-1]
    v1 = v
    max_mask = (v1 >= v0)
    v_max = np.where(max_mask, v1, v0)
    direction[:, :, j] = max_mask
    
    index_mask = (x_range <= j)
    v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
  direction = np.where(mask, direction, 1)
    
  path = np.zeros(value.shape, dtype=np.float32)
  index = mask[:, :, 0].sum(1).astype(np.int64) - 1
  index_range = np.arange(b)
  for j in reversed(range(t_y)):
    path[index_range, index, j] = 1
    index = index + direction[index_range, index, j] - 1
  path = path * mask.astype(np.float32)
  path = torch.from_numpy(path).to(device=device, dtype=dtype)
  return path


def generate_path(duration, mask):
  """
  duration: [b, t_x]
  mask: [b, t_x, t_y]
  """
  device = duration.device
  
  b, t_x, t_y = mask.shape
  cum_duration = torch.cumsum(duration, 1)
  path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:,:-1]
  path = path * mask
  return path


def segment(x: torch.tensor, segment_indices: torch.tensor, segment_size=4, pad_short=False):
    """Segment each sample in a batch based on the provided segment indices

    Args:
        x (torch.tensor): Input tensor.
        segment_indices (torch.tensor): Segment indices.
        segment_size (int): Expected output segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.
    """
    # pad the input tensor if it is shorter than the segment size
    if pad_short and x.shape[-1] < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - x.size(2)))

    segments = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        x_i = x[i]
        if pad_short and index_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (index_end + 1) - x.size(2)))
        segments[i] = x_i[:, index_start:index_end]
    return segments

def rand_segments(
    x: torch.tensor, x_lengths: torch.tensor = None, segment_size=4, let_short_samples=False, pad_short=False
):
    """Create random segments based on the input lengths.

    Args:
        x (torch.tensor): Input tensor.
        x_lengths (torch.tensor): Input lengths.
        segment_size (int): Expected output segment size.
        let_short_samples (bool): Allow shorter samples than the segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.

    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    _x_lenghts = x_lengths.clone()
    B, _, T = x.size()
    if pad_short:
        if T < segment_size:
            x = torch.nn.functional.pad(x, (0, segment_size - T))
            T = segment_size
    if _x_lenghts is None:
        _x_lenghts = T
    len_diff = _x_lenghts - segment_size
    if let_short_samples:
        _x_lenghts[len_diff < 0] = segment_size
        len_diff = _x_lenghts - segment_size
    else:
        assert all(
            len_diff > 0
        ), f" [!] At least one sample is shorter than the segment size ({segment_size}). \n {_x_lenghts}"
    segment_indices = (torch.rand([B]).type_as(x) * (len_diff + 1)).long()
    ret = segment(x, segment_indices, segment_size, pad_short=pad_short)
    return ret, segment_indices

class Adam():
  def __init__(self, params, scheduler, dim_model, warmup_steps=4000, lr=1e0, betas=(0.9, 0.98), eps=1e-9):
    self.params = params
    self.scheduler = scheduler
    self.dim_model = dim_model
    self.warmup_steps = warmup_steps
    self.lr = lr
    self.betas = betas
    self.eps = eps

    self.step_num = 1
    self.cur_lr = lr * self._get_lr_scale()
    
    self._optim = torch.optim.Adam(params, lr=self.cur_lr, betas=betas, eps=eps)

  def _get_lr_scale(self):
    if self.scheduler == "noam":
      return np.power(self.dim_model, -0.5) * np.min([np.power(self.step_num, -0.5), self.step_num * np.power(self.warmup_steps, -1.5)])
    else:
      return 1

  def _update_learning_rate(self):
    self.step_num += 1
    if self.scheduler == "noam":
      self.cur_lr = self.lr * self._get_lr_scale()
      for param_group in self._optim.param_groups:
        param_group['lr'] = self.cur_lr

  def get_lr(self):
    return self.cur_lr

  def step(self):
    self._optim.step()
    self._update_learning_rate()

  def zero_grad(self):
    self._optim.zero_grad()

  def load_state_dict(self, d):
    self._optim.load_state_dict(d)

  def state_dict(self):
    return self._optim.state_dict()


class TacotronSTFT(nn.Module):
  def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = n_mel_channels
    self.sampling_rate = sampling_rate
    self.stft_fn = STFT(filter_length, hop_length, win_length)
    mel_basis = librosa_mel_fn(
        sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

  def mel_spectrogram(self, y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)

    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output)
    return mel_output


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type

    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm


def squeeze(x, x_mask=None, n_sqz=2):
  b, c, t = x.size()

  t = (t // n_sqz) * n_sqz
  x = x[:,:,:t]
  x_sqz = x.view(b, c, t//n_sqz, n_sqz)
  x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c*n_sqz, t//n_sqz)
  
  if x_mask is not None:
    x_mask = x_mask[:,:,n_sqz-1::n_sqz]
  else:
    x_mask = torch.ones(b, 1, t//n_sqz).to(device=x.device, dtype=x.dtype)
  return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
  b, c, t = x.size()

  x_unsqz = x.view(b, n_sqz, c//n_sqz, t)
  x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c//n_sqz, t*n_sqz)

  if x_mask is not None:
    x_mask = x_mask.unsqueeze(-1).repeat(1,1,1,n_sqz).view(b, 1, t*n_sqz)
  else:
    x_mask = torch.ones(b, 1, t*n_sqz).to(device=x.device, dtype=x.dtype)
  return x_unsqz * x_mask, x_mask


def regulate_len(durations, enc_out, pace=1.0, mel_max_len=None):
    """A function that takes predicted durations per encoded token, and repeats enc_out according to the duration.
    NOTE: durations.shape[1] == enc_out.shape[1]

    Args:
        durations (torch.tensor): A tensor of shape (batch x enc_length) that represents how many times to repeat each
            token in enc_out.
        enc_out (torch.tensor): A tensor of shape (batch x enc_length x enc_hidden) that represents the encoded tokens.
        pace (float): The pace of speaker. Higher values result in faster speaking pace. Defaults to 1.0.
        max_mel_len (int): The maximum length above which the output will be removed. If sum(durations, dim=1) >
            max_mel_len, the values after max_mel_len will be removed. Defaults to None, which has no max length.
    """

    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).floor().long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(torch.nn.functional.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype=dtype, device=enc_out.device)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)

    return enc_rep, dec_lens
