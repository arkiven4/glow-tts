import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

import commons 
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence, cmudict
from text.symbols import symbols


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, output_lengths


"""Multi speaker version"""
class TextMelSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.spk_embeds_path = hparams.spk_embeds_path
        self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

    def _filter_text_len(self):
      audiopaths_sid_text_new = []
      for audiopath, sid, text in self.audiopaths_sid_text:
        if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
          audiopaths_sid_text_new.append([audiopath, sid, text])
      self.audiopaths_sid_text = audiopaths_sid_text_new

    def get_mel_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        if self.spk_embeds_path != "":
            filename = audiopath.split("/")[-1].split(".")[0]
            spk_emb = torch.Tensor(np.load(f"{self.spk_embeds_path}/{filename}.npy"))
            return (text, mel, spk_emb)
        else:
            sid = self.get_sid(sid)
            return (text, mel, sid)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.IntTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextMelSpeakerCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            sid[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, output_lengths, sid
    
class TextMelSpeakerEmbedsCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        #sid = torch.LongTensor(len(batch))

        spk_embeds = torch.FloatTensor(len(batch), 512)
        
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            spk_embeds[i, :] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, output_lengths, spk_embeds


"""MyOwn speaker version"""
class TextMelMyOwnLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_lid_text, hparams):
        self.audiopaths_lid_text = load_filepaths_and_text(audiopaths_lid_text)
        print(len(self.audiopaths_lid_text))
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.spk_embeds_path = hparams.spk_embeds_path
        self.emo_embeds_path = hparams.emo_embeds_path
        self.f0_embeds_path = hparams.f0_embeds_path
        self.database_name_index = hparams.database_name_index
        self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_lid_text)

    def _filter_text_len(self):
      audiopaths_lid_text_new = []
      lengths = []
      for now in self.audiopaths_lid_text: 
        if len(now) > 3:
            print(now)

      for audiopath, sid, text in self.audiopaths_lid_text:
        if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
          audiopaths_lid_text_new.append([audiopath, sid, text])
          lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
      self.audiopaths_lid_text = audiopaths_lid_text_new
      self.lengths = lengths

    def get_mel_text_speaker_pair(self, audiopath_lid_text):
        # separate filename, speaker_id and text
        audiopath, lid, text = audiopath_lid_text[0], audiopath_lid_text[1], audiopath_lid_text[2]
        filename = audiopath.split("/")[-1].split(".")[0]
        database_name = audiopath.split("/")[self.database_name_index]

        text = self.get_text(text, lid)
        mel, energy = self.get_mel(audiopath)
        energy = energy[:, :mel.size(1)]
        energy[energy == 0] = torch.nan
        energy = F.pad(input=torch.diff(energy), pad=(0, 1), mode='constant', value=0)
        energy = torch.nan_to_num(energy, nan=0.0)

        spk_emb = torch.Tensor(np.load(f"{self.spk_embeds_path.replace('dataset_name', database_name)}/{filename}.npy"))
        emo_emb = torch.Tensor(np.load(f"{self.emo_embeds_path.replace('dataset_name', database_name)}/{filename}.npy"))

        f0 = np.load(f"{self.f0_embeds_path.replace('dataset_name', database_name)}/{filename}.npy")
        f0 = torch.from_numpy(f0)[None]
        f0[f0 == 0] = torch.nan
        f0 = F.pad(input=torch.diff(f0), pad=(0, 1), mode='constant', value=0)
        f0 = torch.nan_to_num(f0, nan=0.0)
        
        lid = self.get_lid(lid)

        return (text, mel, spk_emb, emo_emb, f0, energy, lid) # f0 or emo

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec, energy = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            energy = torch.squeeze(energy, 0).unsqueeze(0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec, energy

    def get_text(self, text, lang):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, [self.text_cleaners[int(lang)]])
            
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def get_lid(self, lid):
        lid = torch.IntTensor([int(lid)])
        return lid

    def __getitem__(self, index):
        try:
            self.get_mel_text_speaker_pair(self.audiopaths_lid_text[index])
        except:
            print(self.get_mel_text_speaker_pair(self.audiopaths_lid_text[index]))
        return self.get_mel_text_speaker_pair(self.audiopaths_lid_text[index])

    def __len__(self):
        return len(self.audiopaths_lid_text)
    
class TextMelMyOwnCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        #sid = torch.LongTensor(len(batch))

        spk_embeds = torch.FloatTensor(len(batch), 512)
        emo_embeds = torch.FloatTensor(len(batch), 1024)
        #emo_coord = torch.FloatTensor(len(batch), 3)
        lid = torch.LongTensor(len(batch))

        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()

        energy_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        energy_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            spk_embeds[i, :] = batch[ids_sorted_decreasing[i]][2]
            #emo_coord[i, :] = batch[ids_sorted_decreasing[i]][3]
            emo_embeds[i, :] = batch[ids_sorted_decreasing[i]][3]

            f0 = batch[ids_sorted_decreasing[i]][4]
            f0_padded[i, :, :f0.size(1)] = f0

            energy = batch[ids_sorted_decreasing[i]][5]
            energy_padded[i, :, :energy.size(1)] = energy

            lid[i] = batch[ids_sorted_decreasing[i]][6]


        return text_padded, input_lengths, mel_padded, output_lengths, spk_embeds, emo_embeds, f0_padded, energy_padded, lid
    

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True, weights=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.weights = weights
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size