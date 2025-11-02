#!/usr/bin/env python3
"""
Script principal d'entraînement pour les modèles de reconnaissance d'émotions.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --resume checkpoints/last.pt
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import random
import numpy as np

from src.data import create_train_val_loaders
from src.models.emotion_net import EmotionNetNano
from src.training import EmotionTrainer, MetricsCalculator
from src.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LRSchedulerCallback,
    TensorBoardLogger,
    ProgressCallback
)
from src.utils.config import load_config, print_config
from src.utils.visualization import plot_training_history


def set_seed(seed: int = 42):
    """Définit le seed pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(model_config: dict, num_classes: int = 7):
    """
    Crée le modèle selon la configuration.

    Args:
        model_config: Configuration du modèle
        num_classes: Nombre de classes

    Returns:
        Modèle PyTorch
    """
    model_name = model_config.get('name', 'emotionnet_nano').lower()

    if model_name == 'emotionnet_nano':
        model = EmotionNetNano(num_classes=num_classes)
        print(f"✅ Created EmotionNet Nano")

    elif model_name.startswith('resnet'):
        # Import ResNet
        try:
            from src.models.resnet_fer import create_resnet_fer
            model = create_resnet_fer(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=model_config.get('pretrained', False)
            )
            print(f"✅ Created {model_name}")
        except ImportError:
            print(f"❌ ResNet not implemented yet, using EmotionNet Nano")
            model = EmotionNetNano(num_classes=num_classes)

    else:
        print(f"⚠️  Unknown model {model_name}, using EmotionNet Nano")
        model = EmotionNetNano(num_classes=num_classes)

    return model


def create_optimizer(model, optimizer_config: dict):
    """
    Crée l'optimizer selon la configuration.

    Args:
        model: Modèle PyTorch
        optimizer_config: Configuration optimizer

    Returns:
        Optimizer PyTorch
    """
    opt_type = optimizer_config.get('type', 'adamw').lower()

    if opt_type == 'adamw':
        params = optimizer_config.get('adamw', {})
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.get('lr', 0.001),
            betas=params.get('betas', [0.9, 0.999]),
            weight_decay=params.get('weight_decay', 0.0001),
            eps=params.get('eps', 1e-8)
        )

    elif opt_type == 'adam':
        params = optimizer_config.get('adam', {})
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.get('lr', 0.001),
            betas=params.get('betas', [0.9, 0.999]),
            weight_decay=params.get('weight_decay', 0),
            eps=params.get('eps', 1e-8)
        )

    elif opt_type == 'sgd':
        params = optimizer_config.get('sgd', {})
        optimizer = optim.SGD(
            model.parameters(),
            lr=params.get('lr', 0.1),
            momentum=params.get('momentum', 0.9),
            weight_decay=params.get('weight_decay', 0.0001),
            nesterov=params.get('nesterov', True)
        )

    else:
        print(f"⚠️  Unknown optimizer {opt_type}, using AdamW")
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print(f"✅ Created optimizer: {opt_type}")
    return optimizer


def create_scheduler(optimizer, scheduler_config: dict):
    """
    Crée le learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_config: Configuration scheduler

    Returns:
        Scheduler PyTorch (ou None)
    """
    if not scheduler_config or not scheduler_config.get('type'):
        return None

    sched_type = scheduler_config['type'].lower()

    if sched_type == 'reduce_on_plateau':
        params = scheduler_config.get('reduce_on_plateau', {})
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.5),
            patience=params.get('patience', 5),
            min_lr=params.get('min_lr', 1e-6),
            verbose=params.get('verbose', True)
        )
        monitor = params.get('monitor', 'val_loss')

    elif sched_type == 'cosine':
        params = scheduler_config.get('cosine', {})
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', 50),
            eta_min=params.get('eta_min', 1e-6)
        )
        monitor = None

    elif sched_type == 'step':
        params = scheduler_config.get('step', {})
        scheduler = StepLR(
            optimizer,
            step_size=params.get('step_size', 30),
            gamma=params.get('gamma', 0.1)
        )
        monitor = None

    else:
        print(f"⚠️  Unknown scheduler {sched_type}, no scheduler used")
        return None

    print(f"✅ Created scheduler: {sched_type}")
    return scheduler, monitor


def create_criterion(loss_config: dict, device):
    """
    Crée la loss function.

    Args:
        loss_config: Configuration loss
        device: Device

    Returns:
        Loss function
    """
    loss_type = loss_config.get('type', 'cross_entropy').lower()

    if loss_type == 'cross_entropy':
        params = loss_config.get('cross_entropy', {})
        criterion = nn.CrossEntropyLoss(
            weight=params.get('weight'),
            label_smoothing=params.get('label_smoothing', 0.0)
        )

    else:
        print(f"⚠️  Unknown loss {loss_type}, using CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    criterion = criterion.to(device)
    print(f"✅ Created loss function: {loss_type}")
    return criterion


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Train emotion recognition model")

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data config'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda or cpu)'
    )

    args = parser.parse_args()

    # Load configs
    print(f"\n{'='*70}")
    print(f"Loading Configurations")
    print(f"{'='*70}\n")

    train_config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Override from args
    if args.epochs:
        train_config['training']['epochs'] = args.epochs
    if args.batch_size:
        data_config['dataloader']['batch_size'] = args.batch_size
    if args.lr:
        opt_type = train_config['optimizer']['type']
        train_config['optimizer'][opt_type]['lr'] = args.lr
    if args.device:
        train_config['training']['device'] = args.device

    # Set seed
    seed = train_config['training'].get('random_seed', 42)
    set_seed(seed)
    print(f"✅ Random seed set to {seed}")

    # Device
    device_name = train_config['training'].get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create DataLoaders
    print(f"\n{'='*70}")
    print(f"Creating DataLoaders")
    print(f"{'='*70}\n")

    train_loader, val_loader = create_train_val_loaders(
        dataset_type='fer2013',
        data_dir=data_config['data']['root_dir'],
        batch_size=data_config['dataloader']['batch_size'],
        num_workers=data_config['dataloader']['num_workers'],
        img_size=tuple(data_config['preprocessing']['img_size']),
        grayscale=data_config['preprocessing']['grayscale'],
        use_weighted_sampler=data_config['dataloader']['use_weighted_sampler'],
        pin_memory=data_config['dataloader']['pin_memory']
    )

    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")

    # Create model
    print(f"\n{'='*70}")
    print(f"Creating Model")
    print(f"{'='*70}\n")

    model = create_model(
        train_config['model'],
        num_classes=data_config['emotions']['classes']
    )

    # Create optimizer
    optimizer = create_optimizer(model, train_config['optimizer'])

    # Create criterion
    criterion = create_criterion(train_config['loss'], device)

    # Create scheduler
    scheduler_result = create_scheduler(optimizer, train_config.get('scheduler', {}))
    scheduler = scheduler_result[0] if scheduler_result else None
    scheduler_monitor = scheduler_result[1] if scheduler_result else None

    # Create callbacks
    print(f"\n{'='*70}")
    print(f"Setting up Callbacks")
    print(f"{'='*70}\n")

    callbacks = []

    # Progress callback
    callbacks.append(ProgressCallback(train_config['training']['epochs']))

    # Early stopping
    if train_config['callbacks']['early_stopping']['enabled']:
        early_stop = EarlyStopping(
            monitor=train_config['callbacks']['early_stopping']['monitor'],
            patience=train_config['callbacks']['early_stopping']['patience'],
            mode=train_config['callbacks']['early_stopping']['mode'],
            verbose=True
        )
        callbacks.append(early_stop)
        print("✅ Early Stopping enabled")

    # Model checkpoint
    if train_config['callbacks']['model_checkpoint']['enabled']:
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=train_config['callbacks']['model_checkpoint']['checkpoint_dir'],
            monitor=train_config['callbacks']['model_checkpoint']['monitor'],
            mode=train_config['callbacks']['model_checkpoint']['mode'],
            save_best_only=train_config['callbacks']['model_checkpoint']['save_best_only'],
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        print("✅ Model Checkpoint enabled")

    # LR Scheduler callback
    if scheduler is not None:
        lr_callback = LRSchedulerCallback(scheduler, monitor=scheduler_monitor, verbose=True)
        callbacks.append(lr_callback)
        print("✅ LR Scheduler callback enabled")

    # TensorBoard
    if train_config['callbacks']['tensorboard']['enabled']:
        tb_callback = TensorBoardLogger(
            log_dir=train_config['callbacks']['tensorboard']['log_dir'],
            comment=train_config['callbacks']['tensorboard']['comment']
        )
        callbacks.append(tb_callback)
        print("✅ TensorBoard logging enabled")

    # Create trainer
    print(f"\n{'='*70}")
    print(f"Creating Trainer")
    print(f"{'='*70}\n")

    trainer = EmotionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_mixed_precision=train_config['training']['use_mixed_precision'],
        gradient_accumulation_steps=train_config['training']['gradient_accumulation_steps'],
        max_grad_norm=train_config['training']['max_grad_norm'],
        callbacks=callbacks
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config['training']['epochs'],
        start_epoch=start_epoch
    )

    # Plot training history
    print(f"\nPlotting training history...")
    plot_training_history(
        trainer.history,
        save_path='logs/training_history.png'
    )

    # Export model if configured
    if train_config['export']['to_torchscript']:
        print(f"\nExporting model to TorchScript...")
        model.eval()
        example_input = torch.randn(
            1,
            1 if data_config['preprocessing']['grayscale'] else 3,
            *data_config['preprocessing']['img_size']
        ).to(device)

        scripted_model = torch.jit.trace(model, example_input)
        scripted_model.save(train_config['export']['torchscript_path'])
        print(f"✅ TorchScript model saved to {train_config['export']['torchscript_path']}")

    print(f"\n{'='*70}")
    print(f"Training Complete! 🎉")
    print(f"{'='*70}\n")
    print(f"Best validation loss: {min(trainer.history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(trainer.history['val_acc']):.4f}")
    print(f"Best validation F1: {max(trainer.history['val_f1']):.4f}")
    print(f"\nCheckpoints saved in: checkpoints/")
    print(f"TensorBoard logs: tensorboard --logdir logs/tensorboard")
    print()


if __name__ == '__main__':
    main()
