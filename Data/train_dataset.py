import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


def sample_segment(waveform: torch.Tensor, seg_length: int):
    dif = waveform.size(-1) - seg_length
    if dif < 0:
        waveform = torch.hstack((waveform, torch.zeros(1, abs(dif))))
    elif dif > 0:
        index = torch.randint(low=0, high=dif, size=[1])
        waveform = waveform[:, index: seg_length + index]
    return waveform


class TrainDataset(Dataset):

    def __init__(self, data_dir: str, seg_length: float, sample_rate: int):
        super().__init__()
        train_df = pd.DataFrame(columns=['speaker_id', 'filename', 'system_id', 'class_name', 'filepath', 'target'])
        for data_type in ('LA', 'PA'):
            train_type_data = pd.read_csv(os.path.join(data_dir, data_type, f'ASVspoof2019_{data_type}_cm_protocols',
                                                       f'ASVspoof2019.{data_type}.cm.train.trn.txt'),
                                          sep=" ", header=None)
            train_type_data.columns = ['speaker_id', 'filename', 'system_id', 'null', 'class_name']
            train_type_data.drop(columns=['null'], inplace=True)
            files_dir_path = os.path.join(data_dir, data_type, f'ASVspoof2019_{data_type}_train', 'flac/')
            train_type_data['filepath'] = files_dir_path + train_type_data.filename + '.flac'
            train_type_data['target'] = (train_type_data.class_name == 'spoof').astype('long')
            train_df = pd.concat([train_df, train_type_data], axis=0).reset_index(drop=True)
        train_df = train_df.sample(frac=1)
        self.length = train_df.shape[0]
        self.train_df = train_df
        self.sample_rate = sample_rate
        self.seg_length = round(self.sample_rate * seg_length)
        return

    def __getitem__(self, index) -> T_co:
        row = self.train_df.loc[self.train_df.index[index]]
        label = row['target']
        filepath = row['filepath']
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = sample_segment(waveform, self.seg_length)
        return waveform, label

    def __len__(self):
        return self.length
