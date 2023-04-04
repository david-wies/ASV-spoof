import os

import pandas as pd
import torchaudio
from torch.utils.data.dataset import T_co, Dataset


class TestDataset(Dataset):

    def __init__(self, data_dir: str, sample_rate: int):
        super().__init__()
        test_df = pd.DataFrame(columns=['speaker_id', 'filename', 'system_id', 'class_name', 'filepath', 'target'])
        for data_type in ('LA', 'PA'):
            test_type_data = pd.read_csv(
                os.path.join(data_dir, data_type, f'ASVspoof2019_{data_type}_cm_protocols',
                             f'ASVspoof2019.{data_type}.cm.eval.trl.txt'),
                sep=" ", header=None)
            test_type_data.columns = ['speaker_id', 'filename', 'system_id', 'null', 'class_name']
            test_type_data.drop(columns=['null'], inplace=True)
            files_dir_path = os.path.join(data_dir, data_type, f'ASVspoof2019_{data_type}_eval', 'flac/')
            test_type_data['filepath'] = files_dir_path + test_type_data.filename + '.flac'
            test_type_data['target'] = (test_type_data.class_name == 'spoof').astype('long')
            test_df = pd.concat([test_df, test_type_data], axis=0).reset_index(drop=True)
        test_df = test_df.sample(frac=1)
        self.length = test_df.shape[0]
        self.test_df = test_df
        self.sample_rate = sample_rate
        self.indices = test_df.index
        return

    def __getitem__(self, index) -> T_co:
        row = self.test_df.loc[self.indices[index]]
        label = row['target']
        filepath = row['filepath']
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform, label

    def __len__(self):
        return self.length
