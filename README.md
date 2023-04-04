# ASV-spoof

A DeepLearning classifier to distiguish real speech from fake. There are 2 models implemented: ResNet and RawNet2.

This code was tested on Python 3.10.6 with PyTorch 2.0.0. Other packages can be installed by:
```bash
pip install -r requirements.txt
```
## Train Model

To train a model run the command 
```bash
python train_asv_spoof.py [--logs_path LOGS_PATH] [-ckpt CHECKPOINT] [--max_epochs MAX_EPOCHS] [--seg_length SEG_LENGTH] [--train_n_workers TRAIN_N_WORKERS] [--val_n_workers AL_N_WORKERS] [--batch_size BATCH_SIZE] [--sample_rate SAMPLE_RATE] [-lr LEARNING_RATE] {ResNet,RawNet} data_dir
```
## Evaluate Mode

To train a model run the command 
```bash
python eval_asv_spoof.py [--logs_path LOGS_PATH] [--n_workers N_WORKERS] [--sample_rate SAMPLE_RATE] {ResNet,RawNet} data_dir checkpoint
```
The checkpoints from training already been made are located in `logs/{model_name}`.