import os
import json
import argparse
import math
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp

from data_utils import TextMelMyOwnLoader, TextMelMyOwnCollate
import models
import commons
import utils
from text.symbols import symbols
from Noam_Scheduler import Modified_Noam_Scheduler
                          
global_step = 0
global_tqdm = None

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '65535'

  hps = utils.get_hparams()
  mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))


def train_and_eval(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextMelMyOwnLoader(hps.data.training_files, hps.data)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextMelMyOwnCollate(1)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
  if rank == 0:
    val_dataset = TextMelMyOwnLoader(hps.data.validation_files, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

  generator = models.FlowGenerator(
      n_vocab=len(symbols) + getattr(hps.data, "add_blank", False), 
      out_channels=hps.data.n_mel_channels, 
      #n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)
  
  optimizer_g = torch.optim.AdamW(
      generator.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  
  scheduler = Modified_Noam_Scheduler(
      optimizer= optimizer_g,
      base = hps.train.warmup_steps)

  generator = DDP(generator, device_ids=[rank])
  epoch_str = 1
  global_step = 0

  if hps.train.warm_start:
      generator = utils.warm_start_model(hps.train.warm_start_checkpoint, generator, hps.train.ignored_layer)
  else:
    try:
      _, _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, None, scheduler)
      epoch_str += 1
      optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
      #optimizer_g._update_learning_rate()
      global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
      print(e)
      # if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      #   _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
    
  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, scaler, scheduler, logger, writer)
      evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
      utils.save_checkpoint(generator, optimizer_g, scheduler, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, scaler, scheduler, None, None)

    # scheduler.step()


def train(rank, epoch, hps, generator, optimizer_g, train_loader, scaler, scheduler, logger, writer):
  train_loader.sampler.set_epoch(epoch)
  global global_step

  generator.train()
  for batch_idx, (x, x_lengths, y, y_lengths, speakers, emos) in enumerate(tqdm(train_loader)):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)
    emos = emos.cuda(rank, non_blocking=True)

    # Train Generator
    with autocast(enabled=hps.train.fp16_run):
      (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, g=speakers, emo=emos, gen=False)
      
      with autocast(enabled=False):
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

    scheduler.step()

    optimizer_g.zero_grad()
    scaler.scale(loss_g).backward()
    scaler.unscale_(optimizer_g)
    grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
    scaler.step(optimizer_g)
    scaler.update()
    
    if rank==0:
      if batch_idx % hps.train.log_interval == 0:
        (y_gen, *_), *_ = generator.module(x[:1], x_lengths[:1], g=speakers[:1], emo=emos[:1], gen=True)
        # logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #   epoch, batch_idx * len(x), len(train_loader.dataset),
        #   100. * batch_idx / len(train_loader),
        #   loss_g.item()))
        # logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.param_groups[0]['lr']])
        
        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.param_groups[0]['lr'], "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), 
            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), 
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
      for batch_idx, (x, x_lengths, y, y_lengths, speakers, emos) in enumerate(val_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        emos = emos.cuda(rank, non_blocking=True)
        
        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, g=speakers, emo=emos, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        # if batch_idx % hps.train.log_interval == 0:
        #   logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(x), len(val_loader.dataset),
        #     100. * batch_idx / len(val_loader),
        #     loss_g.item()))
        #   logger.info([x.item() for x in loss_gs])
    
    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))

if __name__ == "__main__":
  main()
