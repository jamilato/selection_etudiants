"""
Trainer pour l'entraînement des modèles de reconnaissance d'émotions.

Features:
- Mixed Precision (FP16) avec torch.cuda.amp pour AMD ROCm
- Support des callbacks (early stopping, checkpointing, etc.)
- Métriques détaillées
- TensorBoard logging
- Gradient accumulation
- Gradient clipping

Bonnes pratiques 2025:
- Utilisation de GradScaler pour mixed precision
- Tracking des métriques avec MetricsCalculator
- Support multi-GPU (optionnel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

from .metrics import MetricsCalculator, print_metrics
from .callbacks import CallbackList, Callback


class EmotionTrainer:
    """
    Trainer pour modèles de reconnaissance d'émotions.

    Supporte:
    - Mixed precision training (FP16) pour AMD ROCm
    - Callbacks personnalisés
    - Métriques détaillées
    - Gradient accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Args:
            model: Modèle PyTorch
            optimizer: Optimizer
            criterion: Loss function
            device: Device (cuda ou cpu)
            use_mixed_precision: Utiliser FP16 mixed precision
            gradient_accumulation_steps: Nombre de steps pour accumuler gradients
            max_grad_norm: Gradient clipping (None pour désactiver)
            callbacks: Liste de callbacks
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # Metrics calculators
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        # Best score (pour early stopping)
        self.best_val_loss = float('inf')

        # Setup callbacks avec model et optimizer
        for callback in self.callbacks.callbacks:
            if hasattr(callback, 'set_model'):
                callback.set_model(self.model, self.optimizer)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Entraîne le modèle pour une epoch.

        Args:
            train_loader: DataLoader d'entraînement
            epoch: Numéro de l'epoch

        Returns:
            Dictionnaire de métriques
        """
        self.model.train()
        self.train_metrics.reset()

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(pbar):
            # Déplacer sur device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Callback batch begin
            self.callbacks.on_batch_begin(batch_idx)

            # Forward pass avec mixed precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass avec gradient accumulation
            if self.use_mixed_precision:
                # Scale loss pour accumulation
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                # Update weights après accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            else:
                # Sans mixed precision
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update metrics
            self.train_metrics.update(
                outputs.detach(),
                labels,
                loss.item() * self.gradient_accumulation_steps
            )

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps
            })

            # Callback batch end
            self.callbacks.on_batch_end(batch_idx)

        # Compute epoch metrics
        metrics = self.train_metrics.compute()

        return metrics

    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Valide le modèle pour une epoch.

        Args:
            val_loader: DataLoader de validation
            epoch: Numéro de l'epoch

        Returns:
            Dictionnaire de métriques
        """
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Update metrics
                self.val_metrics.update(outputs, labels, loss.item())

                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        metrics = self.val_metrics.compute()

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 0
    ):
        """
        Entraîne le modèle pour plusieurs epochs.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            epochs: Nombre total d'epochs
            start_epoch: Epoch de départ (pour resume)
        """
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"Max Grad Norm: {self.max_grad_norm}")
        print(f"Total Epochs: {epochs}")
        print(f"{'='*70}\n")

        # Callback train begin
        self.callbacks.on_train_begin()

        # Training loop
        for epoch in range(start_epoch, epochs):
            # Callback epoch begin
            self.callbacks.on_epoch_begin(epoch)

            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)

            epoch_time = time.time() - epoch_start_time

            # Prepare logs
            logs = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['f1_macro'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_macro'],
                'epoch_time': epoch_time
            }

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])

            # Print metrics
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
            print(f"{'='*70}")
            print_metrics(train_metrics, prefix="Train")
            print_metrics(val_metrics, prefix="Validation")

            # Callback epoch end
            self.callbacks.on_epoch_end(epoch, logs)

            # Check early stopping
            if self._check_early_stopping():
                print(f"\n🛑 Training stopped early at epoch {epoch + 1}")
                break

        # Callback train end
        self.callbacks.on_train_end()

        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}\n")

    def _check_early_stopping(self) -> bool:
        """Vérifie si early stopping est déclenché."""
        for callback in self.callbacks.callbacks:
            if hasattr(callback, 'early_stop') and callback.early_stop:
                return True
        return False

    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, any]:
        """
        Évalue le modèle sur l'ensemble de test.

        Args:
            test_loader: DataLoader de test
            return_predictions: Si True, retourne aussi les prédictions

        Returns:
            Dictionnaire de résultats
        """
        print(f"\n{'='*70}")
        print(f"Evaluating on Test Set")
        print(f"{'='*70}\n")

        self.model.eval()
        test_metrics = MetricsCalculator()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                test_metrics.update(outputs, labels, loss.item())

                if return_predictions:
                    preds = torch.argmax(outputs, dim=1)
                    all_predictions.extend(preds.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

        # Compute metrics
        metrics = test_metrics.compute()
        print_metrics(metrics, prefix="Test")

        # Confusion matrix
        cm = test_metrics.get_confusion_matrix()
        print("\nConfusion Matrix:")
        print(cm)

        # Classification report
        print("\nClassification Report:")
        print(test_metrics.get_classification_report())

        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': test_metrics.get_classification_report()
        }

        if return_predictions:
            results['predictions'] = all_predictions
            results['targets'] = all_targets

        return results

    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        """
        Sauvegarde un checkpoint manuel.

        Args:
            filepath: Chemin du fichier
            epoch: Numéro de l'epoch
            **kwargs: Informations additionnelles
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            **kwargs
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        Charge un checkpoint.

        Args:
            filepath: Chemin du checkpoint
            load_optimizer: Charger aussi l'optimizer

        Returns:
            Numéro de l'epoch
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        epoch = checkpoint.get('epoch', 0)

        print(f"✅ Checkpoint loaded from {filepath} (epoch {epoch})")

        return epoch


if __name__ == '__main__':
    print("Testing EmotionTrainer...")

    # Test simple
    from torch.utils.data import TensorDataset

    # Dummy data
    X = torch.randn(100, 1, 48, 48)
    y = torch.randint(0, 7, (100,))

    train_dataset = TensorDataset(X[:80], y[:80])
    val_dataset = TensorDataset(X[80:], y[80:])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Dummy model
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 7)
    )

    # Trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = EmotionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_mixed_precision=False  # CPU test
    )

    # Train 2 epochs
    trainer.fit(train_loader, val_loader, epochs=2)

    print("\n✅ EmotionTrainer test complete")
