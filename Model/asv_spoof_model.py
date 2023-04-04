from typing import Any, Tuple, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from Modules.get_model import get_model


def accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    correct = torch.sum(predictions == labels)
    return correct / predictions.size(0)


def compute_det_curve(target_scores: np.ndarray, non_target_scores: np.ndarray):
    n_scores = target_scores.size + non_target_scores.size
    all_scores = np.concatenate((target_scores, non_target_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(non_target_scores.size)))

    indices = np.argsort(all_scores)
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    non_target_trial_sums = non_target_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    false_rejection_rate = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    false_acceptance_rate = np.concatenate((np.atleast_1d(1), non_target_trial_sums / non_target_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return false_rejection_rate, false_acceptance_rate, thresholds


def compute_eer(target_scores: np.ndarray, non_target_scores: np.ndarray):
    false_rejection_rate, false_acceptance_rate, thresholds = compute_det_curve(target_scores, non_target_scores)
    abs_diffs = np.abs(false_rejection_rate - false_acceptance_rate)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((false_rejection_rate[min_index], false_acceptance_rate[min_index]))
    return eer, thresholds[min_index]


class ASVSpoofModel(pl.LightningModule):

    def __init__(self, model_name: str, learning_rate: float = 0.001, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.model = get_model(model_name, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.validation_logs = list()
        self.model_name = model_name
        self.save_hyperparameters()
        return

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = dict(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            name='RPLateu',
            monitor='val_loss',
            interval='epoch',
            frequency=1,
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def forward(self, waveforms: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        out = self.model(waveforms)
        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        waveforms, labels = batch
        labels = labels.long()
        out = self(waveforms)
        loss = self.loss(out, labels)
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, sync_dist=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        waveforms, labels = batch
        labels = labels.long()
        output = self(waveforms)
        val_loss = self.loss(output, labels)
        predictions = torch.argmax(output, dim=1)
        acc = accuracy(predictions, labels)
        eer, threshold = compute_eer(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_eer', torch.as_tensor(eer), prog_bar=True, sync_dist=True)
        self.log('val_threshold', torch.as_tensor(threshold), sync_dist=True)
        loss = {'val_loss': val_loss, "n_correct_pred": acc, "n_pred": waveforms.size(0), 'eer': eer}
        self.validation_logs.append(loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        waveforms, labels = batch
        labels = labels.long()
        output = self(waveforms)
        test_loss = self.loss(output, labels)
        predictions = torch.argmax(output, dim=1)
        acc = accuracy(predictions, labels)
        eer, threshold = compute_eer(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())
        self.log('test_loss', test_loss, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_eer', torch.as_tensor(eer), prog_bar=True, sync_dist=True)
        loss = {'test_loss': test_loss, "n_correct_pred": acc, "n_pred": waveforms.size(0)}
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        waveforms = batch
        output = self(waveforms)
        predictions = torch.argmax(output, dim=1)
        return predictions
