import torch
import torchaudio
# import librosa
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from torchaudio.transforms import MelSpectrogram

class CynthiaDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, transform=None):
        # if transform is passed, convert raw waveform to spectrogram
        self.audio_labels = pd.read_csv(metadata_file)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.loc[idx, "filename"]+".wav")
        feature, sample_rate = torchaudio.load(audio_path)
        d = {"major":1, "minor":0}
        label = torch.tensor(d[self.audio_labels.loc[idx, "scale"]])
        if self.transform:
            feature = self.transform(feature)
        return feature, label

if __name__=="__main__":
    transform = MelSpectrogram(sample_rate=16000, n_mels=32)
    dataset = CynthiaDataset(metadata_sine_test, sine_test_path, transform)

    subset_indices = [0,1]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    data = DataLoader(subset, batch_size=1, num_workers=3, shuffle=False)

    for idx, data in enumerate(data):
        features, labels = data
        print(len(features))