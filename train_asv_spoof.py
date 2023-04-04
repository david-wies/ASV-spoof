import os.path
from argparse import ArgumentParser
from typing import Union

from IPython import embed
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from Data.data_module import DataModule
from Model.asv_spoof_model import ASVSpoofModel

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logs_path', help='Path to save the logs', default='logs')
    parser.add_argument('-ckpt', '--checkpoint', help='Path to checkpoint', default=None)
    parser.add_argument('--max_epochs', type=int, default=None, help='The maximum number of epoch to train')
    parser.add_argument('model_name', help='The name of the model to use', choices=['ResNet', 'RawNet'])
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument('--seg_length', type=float, default=1.5, help='The length, in seconds, of audio to use')
    parser.add_argument('--train_n_workers', type=int, default=10,
                        help='The number of workers to use for train dataloader')
    parser.add_argument('--val_n_workers', type=int, default=2,
                        help='The number of workers to use for validation dataloader')
    parser.add_argument('--batch_size', type=int, default=5, help='The size of the batch for the train data')
    parser.add_argument('--sample_rate', type=int, default=16000, help='The sample rate of the audio')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='The learning rate to use')
    args = parser.parse_args()
    asv_model = ASVSpoofModel(args.model_name, args.learning_rate)

    data_module = DataModule(args.data_dir, args.batch_size, args.seg_length, args.train_n_workers, args.val_n_workers,
                             args.sample_rate)

    data_module.setup('init')

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    logs_path = os.path.join(args.logs_path, asv_model.model_name)

    trainer = Trainer(accelerator='auto',
                      max_epochs=args.max_epochs,
                      callbacks=[TQDMProgressBar(refresh_rate=20),
                                 ModelCheckpoint(dirpath=os.path.join(logs_path, 'ckpt'), save_top_k=3,
                                                 monitor='val_loss'),
                                 LearningRateMonitor(logging_interval='epoch')
                                 ],
                      logger=[
                          TensorBoardLogger(save_dir=logs_path, sub_dir='TensorBoardLogs'),
                          CSVLogger(save_dir=logs_path)
                      ],
                      default_root_dir=logs_path,
                      )
    trainer.fit(asv_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path=args.checkpoint)

    embed()
