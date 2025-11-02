"""
Point d'entrée principal du système d'identification d'étudiants avec analyse d'émotions
Optimisé pour AMD Radeon 7900 XT avec ROCm
"""

import argparse
import sys
import cv2
import torch
import yaml
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def load_config(config_path):
    """Charge la configuration depuis un fichier YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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

    # TODO: Implémenter le système complet
    # Pour l'instant, juste un test de webcam

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Erreur: Impossible d'ouvrir la webcam")
        return

    print("✅ Webcam détectée")
    print("Appuyez sur 'q' pour quitter")

    fps_target = config['realtime']['fps_target']
    show_fps = config['realtime']['show_fps']

    import time
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Erreur de lecture de la frame")
            break

        # Calculer FPS
        if show_fps:
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Afficher FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Afficher instructions
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(config['visualization']['window_name'], frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Mode temps réel terminé")


def main_video(config, video_path):
    """Mode traitement vidéo"""
    print(f"\n🎬 Traitement de la vidéo: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")

    # TODO: Implémenter le traitement vidéo complet

    cap.release()
    print("✅ Traitement vidéo terminé")


def main_image(config, image_path):
    """Mode image unique"""
    print(f"\n🖼️  Traitement de l'image: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Erreur: Impossible de lire l'image {image_path}")
        return

    print(f"   Dimensions: {image.shape}")

    # TODO: Implémenter le traitement d'image complet

    print("✅ Traitement image terminé")


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
            main_video(config, args.input)

        elif args.mode == 'image':
            if not args.input:
                print("❌ Erreur: --input requis pour le mode image")
                sys.exit(1)
            main_image(config, args.input)

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
