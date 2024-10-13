import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import gc

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['model']

    random_weight_layer = []
    mismatched_layers = []
    unfound_layers = []
    
    for key, value in model_dict.items(): # model_dict warmstart weight
        if hasattr(model, 'module'): # model is current model
            if key in model.module.state_dict() and value.size() != model.module.state_dict()[key].size():
                try:
                    model_dict[key] = transfer_weight(model_dict[key], model.module.state_dict()[key].size())
                    if model_dict[key].size() != model.module.state_dict()[key].size():
                      mismatched_layers.append(key)
                    else:
                      random_weight_layer.append(key)
                except:
                    mismatched_layers.append(key)
        else:
            if key in model.state_dict() and value.size() != model.state_dict()[key].size():
                try:
                    model_dict[key] = transfer_weight(model_dict[key], model.state_dict()[key].size())
                    if model_dict[key].size() != model.state_dict()[key].size():
                      mismatched_layers.append(key)
                    else:
                      random_weight_layer.append(key)
                except:
                    mismatched_layers.append(key)
    
    # for key, value in model_dict.items():
    #   if hasattr(model, 'module'):
    #     if key not in model.module.state_dict():
    #       unfound_layers.append(key)
    #   else:
    #     if key not in model.state_dict():
    #       unfound_layers.append(key)
        
    print("Mismatched")
    print(mismatched_layers)

    print("random_weight_layer")
    print(random_weight_layer)
    
    ignore_layers = ignore_layers + mismatched_layers
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        if hasattr(model, 'module'):
          dummy_dict = model.module.state_dict()
          dummy_dict.update(model_dict)
        else:
          dummy_dict = model.state_dict()
          dummy_dict.update(model_dict)
        model_dict = dummy_dict

    if hasattr(model, 'module'):
      model.module.load_state_dict(model_dict, strict=False)
    else:
      model.load_state_dict(model_dict, strict=False)
    
    #del checkpoint_dict, model_dict, dummy_dict
    #gc.collect()
    #torch.cuda.empty_cache()
    return model

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = 1
  if 'iteration' in checkpoint_dict.keys():
    iteration = checkpoint_dict['iteration']
  if 'learning_rate' in checkpoint_dict.keys():
    learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  # if scheduler is not None and 'scheduler' in checkpoint_dict.keys():
  #   scheduler.load_state_dict(checkpoint_dict['scheduler'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))

  return model, optimizer, scheduler, learning_rate, iteration


def save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_pitch_to_numpy(pitch, pitch_pred):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots()
  ax.plot(pitch, label="Original")
  ax.plot(pitch_pred, label="Prediction")
  plt.tight_layout()
  plt.legend(loc="upper left")

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots()
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment, aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f ] #if int(line.strip().split(split)[1]) == 0 or int(line.strip().split(split)[1]) == 2
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
  
def transfer_weight(original_tensor, target_size):
    differences = [target_size[i] - original_tensor.size(i) for i in range(len(target_size))]
    for i, diff in enumerate(differences):
        if diff > 0:
            new_dims = list(original_tensor.size())
            new_dims[i] = diff
            rand_weight = torch.randn(*new_dims)
            original_tensor = torch.cat([original_tensor, rand_weight], dim=i)
        # elif diff < 0:
        #     slices = []
        #     for j in range(len(target_size)):
        #         if j == i:
        #             slices.append(slice(0, original_tensor.size(j) + diff))
        #         else:
        #             slices.append(slice(0, original_tensor.size(j)))
        #     slices[i] = slice(0, target_size[i])
        #     original_tensor = original_tensor[slices]

    return original_tensor
