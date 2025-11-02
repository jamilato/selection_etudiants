"""
Callbacks pour l'entraînement.

Implémente:
- EarlyStopping (arrête l'entraînement si pas d'amélioration)
- ModelCheckpoint (sauvegarde les meilleurs modèles)
- LRSchedulerCallback (ajuste learning rate)
- TensorBoardLogger (logging TensorBoard)
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class Callback:
    """Classe de base pour les callbacks."""

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Appelé au début de l'entraînement."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Appelé à la fin de l'entraînement."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Appelé au début de chaque epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Appelé à la fin de chaque epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Appelé au début de chaque batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Appelé à la fin de chaque batch."""
        pass


class EarlyStopping(Callback):
    """
    Early Stopping callback.

    Arrête l'entraînement si la métrique surveillée ne s'améliore pas
    pendant patience epochs.

    Attributes:
        monitor: Métrique à surveiller (ex: 'val_loss')
        patience: Nombre d'epochs sans amélioration avant arrêt
        mode: 'min' (loss) ou 'max' (accuracy)
        min_delta: Changement minimum pour considérer une amélioration
        verbose: Si True, affiche des messages
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Args:
            monitor: Métrique à surveiller
            patience: Patience (nombre d'epochs)
            mode: 'min' pour minimiser, 'max' pour maximiser
            min_delta: Delta minimum pour amélioration
            verbose: Afficher messages
        """
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        # État
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        # Pour mode
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
            self.best_score = float('inf')
        else:  # max
            self.monitor_op = lambda x, y: x > y + min_delta
            self.best_score = float('-inf')

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Vérifie si early stopping doit être déclenché."""
        if logs is None:
            return

        current_score = logs.get(self.monitor)

        if current_score is None:
            if self.verbose:
                print(f"Warning: {self.monitor} not found in logs")
            return

        # Vérifier amélioration
        if self.monitor_op(current_score, self.best_score):
            # Amélioration
            self.best_score = current_score
            self.counter = 0

            if self.verbose:
                print(f"  EarlyStopping: {self.monitor} improved to {current_score:.4f}")

        else:
            # Pas d'amélioration
            self.counter += 1

            if self.verbose:
                print(f"  EarlyStopping: {self.monitor} did not improve "
                      f"(patience: {self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")


class ModelCheckpoint(Callback):
    """
    Model Checkpoint callback.

    Sauvegarde le modèle à chaque amélioration de la métrique surveillée.

    Attributes:
        checkpoint_dir: Dossier pour sauvegarder les checkpoints
        monitor: Métrique à surveiller
        mode: 'min' ou 'max'
        save_best_only: Si True, sauvegarde seulement le meilleur modèle
        verbose: Afficher messages
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        """
        Args:
            checkpoint_dir: Dossier des checkpoints
            monitor: Métrique à surveiller
            mode: 'min' ou 'max'
            save_best_only: Sauvegarder seulement le meilleur
            save_weights_only: Sauvegarder seulement les poids (pas optimizer)
            verbose: Afficher messages
        """
        super().__init__()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        # Best score
        if mode == 'min':
            self.best_score = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best_score = float('-inf')
            self.monitor_op = lambda x, y: x > y

        # Pour stocker model et optimizer
        self.model = None
        self.optimizer = None

    def set_model(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Définit le modèle et optimizer à sauvegarder."""
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Sauvegarde le modèle si amélioration."""
        if logs is None or self.model is None:
            return

        current_score = logs.get(self.monitor)

        if current_score is None:
            if self.verbose:
                print(f"Warning: {self.monitor} not found in logs")
            return

        # Vérifier si on doit sauvegarder
        should_save = False

        if self.save_best_only:
            if self.monitor_op(current_score, self.best_score):
                self.best_score = current_score
                should_save = True
        else:
            should_save = True

        if should_save:
            self._save_checkpoint(epoch, current_score)

    def _save_checkpoint(self, epoch: int, score: float):
        """Sauvegarde un checkpoint."""
        # Nom du fichier
        if self.save_best_only:
            filename = f"best_model_{self.monitor}.pt"
        else:
            filename = f"checkpoint_epoch{epoch+1:03d}_{self.monitor}={score:.4f}.pt"

        filepath = self.checkpoint_dir / filename

        # Créer checkpoint
        if self.save_weights_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                self.monitor: score
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                self.monitor: score
            }

        # Sauvegarder
        torch.save(checkpoint, filepath)

        if self.verbose:
            print(f"  💾 Model saved to {filepath}")


class LRSchedulerCallback(Callback):
    """
    Learning Rate Scheduler callback.

    Ajuste le learning rate selon un scheduler.
    """

    def __init__(
        self,
        scheduler: _LRScheduler,
        monitor: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            scheduler: PyTorch LR scheduler
            monitor: Métrique à surveiller (pour ReduceLROnPlateau)
            verbose: Afficher messages
        """
        super().__init__()

        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Step le scheduler."""
        # ReduceLROnPlateau nécessite une métrique
        if hasattr(self.scheduler, 'step') and self.monitor:
            if logs and self.monitor in logs:
                metric_value = logs[self.monitor]
                self.scheduler.step(metric_value)

                if self.verbose:
                    current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                    print(f"  Learning rate: {current_lr:.6f}")
        else:
            # Autres schedulers
            self.scheduler.step()

            if self.verbose:
                current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                print(f"  Learning rate: {current_lr:.6f}")


class TensorBoardLogger(Callback):
    """
    TensorBoard Logger callback.

    Log les métriques dans TensorBoard.
    """

    def __init__(
        self,
        log_dir: str = 'logs/tensorboard',
        comment: str = ''
    ):
        """
        Args:
            log_dir: Dossier pour les logs TensorBoard
            comment: Commentaire pour ce run
        """
        super().__init__()

        from torch.utils.tensorboard import SummaryWriter

        # Créer dossier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{timestamp}_{comment}" if comment else timestamp

        self.log_dir = Path(log_dir) / run_name
        self.writer = SummaryWriter(str(self.log_dir))

        print(f"📊 TensorBoard logging to: {self.log_dir}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log les métriques."""
        if logs is None:
            return

        for metric_name, metric_value in logs.items():
            if isinstance(metric_value, (int, float)):
                self.writer.add_scalar(metric_name, metric_value, epoch)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Ferme le writer."""
        self.writer.close()


class ProgressCallback(Callback):
    """
    Callback pour afficher la progression.
    """

    def __init__(self, total_epochs: int):
        """
        Args:
            total_epochs: Nombre total d'epochs
        """
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Affiche le début de l'epoch."""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{self.total_epochs}")
        print(f"{'='*70}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Affiche les métriques de fin d'epoch."""
        if logs:
            metrics_str = "  ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in logs.items()
                if k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1']
            ])
            print(f"\n{metrics_str}")


class CallbackList:
    """
    Liste de callbacks.

    Permet de gérer plusieurs callbacks ensemble.
    """

    def __init__(self, callbacks: list):
        """
        Args:
            callbacks: Liste de callbacks
        """
        self.callbacks = callbacks or []

    def on_train_begin(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


if __name__ == '__main__':
    # Test des callbacks
    print("Testing callbacks...")

    # Test EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True)

    for epoch in range(10):
        # Simuler logs
        logs = {'val_loss': 1.0 - epoch * 0.05 if epoch < 5 else 0.75}

        early_stopping.on_epoch_end(epoch, logs)

        if early_stopping.early_stop:
            print(f"Training stopped at epoch {epoch + 1}")
            break

    print("\n✅ Callbacks test complete")
