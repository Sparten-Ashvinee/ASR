import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

from data_augment import aug_effects, audiomentations_lib

class UrbanSoundDataset(Dataset):

    def __init__(self,
                 filenames,
                 labels,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        #self.annotations = pd.read_csv(annotations_file)
        self.filenames = filenames
        self.labels = labels
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation    #.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #print('index: ',index)
        audio_sample_path = self.filenames[index]
        label = self.labels[index]
        #print(1)
        signal, sr = torchaudio.load(audio_sample_path)
        # sample = signal.numpy()
        # signal = augment(sample, sample_rate=self.target_sample_rate)
        #effects = aug_effects(self.target_sample_rate)
        #signal, sr = torchaudio.sox_effects.apply_effects_tensor(signal, self.target_sample_rate, effects)
        #signal = torch.tensor(signal)        #.to(self.device)
        #print(2)
        signal = self._resample_if_necessary(signal, sr)
        #print(3)
        signal = self._mix_down_if_necessary(signal)
        #print(4)
        signal = self._cut_if_necessary(signal)
        #print(5)
        signal = self._right_pad_if_necessary(signal)
        #print(6)
        signal = self.transformation(signal)
        #print('Signal: ', signal)
        #print('label: ', label)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"{self.annotations.iloc[index, 1]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        #print(path)
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 1]
        class_mapping = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        label = class_mapping.index(label)
        return label
        

def get_data(annotations_file, audio_dir):

  annotations = pd.read_csv(annotations_file)

  fn = []
  lb = []
  for i in range(len(annotations)):
    fold = annotations.iloc[i, 1]
    file = annotations.iloc[i, 0]
    filename = os.path.join(audio_dir, fold, file)
    fn.append(filename)

    label = annotations.iloc[i, 1]
    class_mapping = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    label = class_mapping.index(label)
    lb.append(label)

  return fn, lb

def clear_file(ANNOTATIONS_FILE, DIR, new_annpath):
    anns = pd.read_csv(ANNOTATIONS_FILE)
    fn = anns['filename'].tolist()
    emm = anns['emotion'].tolist()

    dl = []
    count=0
    for item in fn:
      if item.split('.')[-1] != 'mp3':
          count+=1
          dl.append(anns[anns['filename']==item].index.values[0])

    print('Number of other files than mp3 detected: ', count)
    anns = anns.drop(index=dl)
    anns.reset_index(inplace=True, drop=True)
    anns.to_csv(new_annpath, index=False)
    
    return new_annpath
    
    
    
    
    
    
    
    
    
