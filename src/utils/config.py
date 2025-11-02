"""
Utilitaires pour charger et gérer les configurations YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge un fichier de configuration YAML.

    Args:
        config_path: Chemin vers le fichier YAML

    Returns:
        Dictionnaire de configuration

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        yaml.YAMLError: Si erreur de parsing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_all_configs(config_dir: str = 'configs') -> Dict[str, Dict]:
    """
    Charge tous les fichiers de configuration.

    Args:
        config_dir: Dossier contenant les configs

    Returns:
        Dictionnaire {nom: config}
    """
    config_dir = Path(config_dir)

    configs = {}

    # Charger les configs principales
    config_files = {
        'data': 'data_config.yaml',
        'train': 'train_config.yaml',
        'model': 'model_config.yaml',
    }

    for name, filename in config_files.items():
        filepath = config_dir / filename

        if filepath.exists():
            configs[name] = load_config(filepath)
        else:
            print(f"Warning: {filename} not found")

    return configs


def save_config(config: Dict[str, Any], config_path: str):
    """
    Sauvegarde une configuration en YAML.

    Args:
        config: Dictionnaire de configuration
        config_path: Chemin de sauvegarde
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config saved to {config_path}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge deux configurations (override a priorité).

    Args:
        base_config: Config de base
        override_config: Config qui override

    Returns:
        Config mergée
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_config_value(config: Dict, path: str, default: Any = None) -> Any:
    """
    Récupère une valeur dans une config imbriquée.

    Args:
        config: Configuration
        path: Chemin (ex: "training.optimizer.lr")
        default: Valeur par défaut

    Returns:
        Valeur trouvée ou default
    """
    keys = path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_config_value(config: Dict, path: str, value: Any):
    """
    Définit une valeur dans une config imbriquée.

    Args:
        config: Configuration
        path: Chemin (ex: "training.epochs")
        value: Valeur à définir
    """
    keys = path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def update_config_from_args(config: Dict, args: Dict) -> Dict:
    """
    Met à jour une config depuis des arguments CLI.

    Args:
        config: Configuration de base
        args: Arguments (dict ou argparse.Namespace)

    Returns:
        Config mise à jour
    """
    if hasattr(args, '__dict__'):
        args = vars(args)

    updated_config = config.copy()

    # Mapping des arguments CLI vers config paths
    arg_mappings = {
        'epochs': 'training.epochs',
        'batch_size': 'dataloader.batch_size',
        'lr': 'optimizer.adamw.lr',
        'device': 'training.device',
        'checkpoint_dir': 'callbacks.model_checkpoint.checkpoint_dir',
    }

    for arg_name, config_path in arg_mappings.items():
        if arg_name in args and args[arg_name] is not None:
            set_config_value(updated_config, config_path, args[arg_name])

    return updated_config


def print_config(config: Dict, indent: int = 0):
    """
    Affiche une configuration de manière formatée.

    Args:
        config: Configuration
        indent: Niveau d'indentation
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def validate_config(config: Dict, required_keys: list) -> bool:
    """
    Valide qu'une config contient les clés requises.

    Args:
        config: Configuration
        required_keys: Liste de clés requises (support dot notation)

    Returns:
        True si valide

    Raises:
        ValueError: Si clé manquante
    """
    for key_path in required_keys:
        value = get_config_value(config, key_path)

        if value is None:
            raise ValueError(f"Required config key missing: {key_path}")

    return True


if __name__ == '__main__':
    # Test
    print("Testing config utilities...")

    # Load all configs
    try:
        configs = load_all_configs('configs')
        print(f"\n✅ Loaded {len(configs)} config files")

        for name, config in configs.items():
            print(f"\n{name.upper()} Config:")
            print("-" * 40)
            # Print first level keys
            for key in list(config.keys())[:5]:
                print(f"  {key}: ...")

    except Exception as e:
        print(f"❌ Error: {e}")
