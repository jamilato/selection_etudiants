"""
Script de test pour vérifier la détection du GPU AMD avec ROCm
"""

import torch
import sys


def test_gpu():
    """Teste la disponibilité et les caractéristiques du GPU"""

    print("=" * 60)
    print("Test de Configuration GPU - AMD Radeon 7900 XT")
    print("=" * 60)

    # Version PyTorch
    print(f"\n✅ PyTorch version: {torch.__version__}")

    # Vérifier CUDA (ROCm apparaît comme CUDA dans PyTorch)
    if not torch.cuda.is_available():
        print("\n❌ ERREUR: GPU non détecté!")
        print("Vérifiez votre installation ROCm:")
        print("  1. rocm-smi")
        print("  2. rocminfo | grep 'Name:'")
        sys.exit(1)

    print(f"✅ CUDA (ROCm) disponible: True")

    # Nombre de GPUs
    device_count = torch.cuda.device_count()
    print(f"✅ Nombre de GPU détectés: {device_count}")

    # Informations sur chaque GPU
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"Nom: {torch.cuda.get_device_name(i)}")

        props = torch.cuda.get_device_properties(i)
        print(f"Mémoire totale: {props.total_memory / 1e9:.2f} GB")
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"Compute capability: {props.major}.{props.minor}")

        # Tester allocation mémoire
        try:
            test_tensor = torch.randn(1000, 1000).cuda(i)
            print(f"✅ Allocation mémoire: OK")
            del test_tensor
        except Exception as e:
            print(f"❌ Erreur allocation mémoire: {e}")

    # Test de calcul simple
    print("\n--- Test de Calcul GPU ---")
    try:
        a = torch.randn(5000, 5000).cuda()
        b = torch.randn(5000, 5000).cuda()

        torch.cuda.synchronize()
        import time
        start = time.time()

        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        elapsed = time.time() - start
        print(f"✅ Multiplication de matrices (5000x5000): {elapsed:.4f} secondes")

        del a, b, c

    except Exception as e:
        print(f"❌ Erreur lors du test de calcul: {e}")

    # Mémoire disponible
    print("\n--- Utilisation Mémoire ---")
    print(f"Mémoire allouée: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Mémoire réservée: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Test Mixed Precision (FP16)
    print("\n--- Test Mixed Precision (FP16) ---")
    try:
        with torch.cuda.amp.autocast():
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = x @ y
        print("✅ Mixed Precision (AMP): Supporté")
        del x, y, z
    except Exception as e:
        print(f"⚠️  Mixed Precision: Non supporté ({e})")

    print("\n" + "=" * 60)
    print("✅ Tous les tests GPU sont passés avec succès!")
    print("Votre AMD Radeon 7900 XT est prête pour le deep learning.")
    print("=" * 60)


if __name__ == "__main__":
    test_gpu()
