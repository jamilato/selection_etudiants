"""
Point d'entrée principal du système d'identification d'étudiants avec analyse d'émotions
Optimisé pour AMD Radeon 7900 XT avec ROCm
"""

import argparse
import sys
import cv2
import torch
import yaml
import logging
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.core.system import create_system_from_config


def check_gpu():
    """Vérifie la disponibilité du GPU"""
    if not torch.cuda.is_available():
        print("⚠️  WARNING: GPU non détecté. Le système fonctionnera sur CPU (très lent).")
        print("Vérifiez votre installation ROCm avec: rocm-smi")
        return False

    print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return True


def main_realtime(config):
    """Mode temps réel avec webcam"""
    print("\n🎥 Démarrage du mode temps réel...")

    try:
        # Créer le système complet
        print("   Initialisation du système...")
        system = create_system_from_config(config)

        print("✅ Système initialisé avec succès")
        print(f"   - Détecteur: {config['face_detection']['method']}")
        print(f"   - Modèle: {config['emotion_model']['architecture']}")
        print(f"   - Device: {config['device']['type']}")
        print()
        print("Contrôles:")
        print("  - 'q': Quitter")
        print("  - 's': Prendre une capture d'écran")
        print("  - 'p': Pause/Resume")
        print()

        # Lancer la webcam
        system.run_webcam(camera_id=0)

    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du système: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Mode temps réel terminé")


def main_video(config, video_path, output_path=None):
    """Mode traitement vidéo"""
    print(f"\n🎬 Traitement de la vidéo: {video_path}")

    if not Path(video_path).exists():
        print(f"❌ Erreur: Fichier introuvable {video_path}")
        return

    try:
        # Créer le système
        print("   Initialisation du système...")
        system = create_system_from_config(config)

        print("✅ Système initialisé")

        # Traiter la vidéo
        system.process_video(
            video_path=video_path,
            output_path=output_path,
            show_window=True
        )

    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()
        return

    print("✅ Traitement vidéo terminé")


def main_image(config, image_path, output_path=None):
    """Mode image unique"""
    print(f"\n🖼️  Traitement de l'image: {image_path}")

    if not Path(image_path).exists():
        print(f"❌ Erreur: Fichier introuvable {image_path}")
        return

    try:
        # Créer le système
        print("   Initialisation du système...")
        system = create_system_from_config(config)

        print("✅ Système initialisé")

        # Traiter l'image
        results = system.process_image(
            image_path=image_path,
            output_path=output_path
        )

        # Afficher les résultats
        print(f"\n📊 Résultats de l'analyse:")
        print(f"   Visages détectés: {len(results)}")

        for i, result in enumerate(results, 1):
            print(f"\n   Visage #{i}:")
            if 'student_name' in result:
                print(f"     - Étudiant: {result['student_name']}")
                if result.get('student_similarity'):
                    print(f"     - Similarité: {result['student_similarity']:.2f}")
            print(f"     - Émotion: {result['emotion']}")
            print(f"     - Confiance: {result['emotion_confidence']:.2f}")
            print(f"     - Position: {result['bbox']}")

        if output_path:
            print(f"\n✅ Image annotée sauvegardée: {output_path}")

        # Afficher l'image annotée
        if output_path and Path(output_path).exists():
            print("\nAppuyez sur une touche pour fermer l'image...")
            img = cv2.imread(output_path)
            cv2.imshow("Résultat", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Traitement image terminé")


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Système d'Identification d'Étudiants avec Analyse d'Émotions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Mode temps réel (webcam)
  python main.py --mode realtime

  # Traiter une vidéo
  python main.py --mode video --input video.mp4

  # Traiter une image
  python main.py --mode image --input image.jpg

  # Utiliser une configuration personnalisée
  python main.py --config my_config.yaml --mode realtime
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['realtime', 'video', 'image'],
        default='realtime',
        help='Mode d\'exécution (défaut: realtime)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Chemin vers le fichier de configuration (défaut: configs/config.yaml)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Chemin vers l\'image ou vidéo (requis pour modes video/image)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Chemin vers le modèle (override config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device à utiliser (override config)'
    )

    parser.add_argument(
        '--fps-target',
        type=int,
        help='FPS cible pour temps réel (override config)'
    )

    parser.add_argument(
        '--show-fps',
        action='store_true',
        help='Afficher FPS en temps réel'
    )

    parser.add_argument(
        '--save-output',
        type=str,
        help='Sauvegarder la sortie vidéo'
    )

    args = parser.parse_args()

    # Afficher banner
    print("=" * 70)
    print("  Système d'Identification d'Étudiants avec Analyse d'Émotions")
    print("  Optimisé pour AMD Radeon 7900 XT avec ROCm")
    print("  Version 1.0.0")
    print("=" * 70)

    # Configuration du logging
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Charger configuration
    try:
        config = load_config(args.config)
        print(f"\n✅ Configuration chargée depuis: {args.config}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        sys.exit(1)

    # Overrides depuis arguments
    if args.device:
        config['device']['type'] = args.device

    if args.model:
        config['emotion_model']['weights_path'] = args.model

    if args.fps_target:
        config['realtime']['fps_target'] = args.fps_target

    if args.show_fps:
        config['realtime']['show_fps'] = True

    if args.save_output:
        config['output']['save_video'] = True
        config['output']['output_video_path'] = args.save_output

    # Vérifier GPU
    if config['device']['type'] == 'cuda':
        check_gpu()

    # Exécuter selon le mode
    try:
        if args.mode == 'realtime':
            main_realtime(config)

        elif args.mode == 'video':
            if not args.input:
                print("❌ Erreur: --input requis pour le mode video")
                sys.exit(1)
            main_video(config, args.input, output_path=args.save_output)

        elif args.mode == 'image':
            if not args.input:
                print("❌ Erreur: --input requis pour le mode image")
                sys.exit(1)

            # Générer un chemin de sortie automatique si non spécifié
            output_path = args.save_output
            if output_path is None:
                input_path = Path(args.input)
                output_path = str(input_path.parent / f"{input_path.stem}_annotated{input_path.suffix}")

            main_image(config, args.input, output_path=output_path)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  Fin du programme")
    print("=" * 70)


if __name__ == "__main__":
    main()
