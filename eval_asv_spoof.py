import glob
import os
import re
from argparse import ArgumentParser

from IPython import embed
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from Data.data_module import DataModule
from Model.asv_spoof_model import ASVSpoofModel


def get_checkpoint(path: str) -> str:
    """
    :param path: Path to a checkpoint or directory of checkpoints
    :return: Path to the checkpoint that trained the largest number of epochs
    """
    if os.path.isfile(path):
        return path
    epoch = -1
    step = -1
    pattern = re.compile('epoch=(\d+)-step=(\d+)')
    ckpt_path = ''
    for path in glob.iglob(os.path.join(path, 'ckpt', '*.ckpt')):
        name = os.path.splitext(os.path.basename(path))[0]
        match = pattern.match(name)
        current_epoch = int(match.group(1))
        current_step = int(match.group(2))
        if current_epoch > epoch or current_epoch == epoch and current_step >= step:
            ckpt_path = path
            epoch = current_epoch
            step = current_step
    return ckpt_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logs_path', help='Path to save the logs', default='logs')
    parser.add_argument('model_name', help='The name of the model to use', choices=['ResNet', 'RawNet'])
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--n_workers', type=int, default=2, help='The number of workers to use for the dataloader')
    parser.add_argument('--sample_rate', type=int, default=16000, help='The sample rate of the audio')
    args = parser.parse_args()

    asv_spoof_model = ASVSpoofModel(args.model_name)
    data_module = DataModule(args.data_dir, 1, 0.2, args.n_workers, args.n_workers, args.sample_rate)
    data_module.setup('init')
    logs_path = os.path.join(args.logs_path, asv_spoof_model.model_name)
    ckpt = get_checkpoint(args.checkpoint)

    trainer = Trainer(accelerator='auto',
                      max_epochs=1,
                      devices=args.devices,
                      callbacks=TQDMProgressBar(refresh_rate=20),
                      logger=[
                          TensorBoardLogger(save_dir=logs_path, sub_dir='TensorBoardLogs'),
                          CSVLogger(save_dir=logs_path)
                      ],
                      default_root_dir=logs_path
                      )

    test_results = trainer.test(asv_spoof_model.eval(), dataloaders=data_module.test_dataloader(), ckpt_path=ckpt)

    embed()
