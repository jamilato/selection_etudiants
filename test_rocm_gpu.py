#!/usr/bin/env python3
"""
Script de test complet pour ROCm + PyTorch GPU
Vérifie que le GPU AMD est détecté et fonctionnel
"""

import sys
import time

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def test_pytorch():
    """Test PyTorch installation et GPU"""
    print_header("1. Test PyTorch")

    try:
        import torch
        print(f"✓ PyTorch installé: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch non installé")
        print("  Installez avec: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2")
        return False

    # Vérifier CUDA (ROCm utilise l'API CUDA)
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA/ROCm disponible: {cuda_available}")

    if not cuda_available:
        print("\n⚠ GPU non détecté. Raisons possibles:")
        print("  1. ROCm pas installé : sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms")
        print("  2. WSL pas redémarré : wsl --shutdown (dans PowerShell)")
        print("  3. Variables d'environnement manquantes : source ~/.bashrc")
        print("  4. PyTorch CPU installé au lieu de ROCm")
        return False

    # Informations GPU
    print(f"\n✓ Nombre de GPUs: {torch.cuda.device_count()}")
    print(f"✓ GPU actuel: {torch.cuda.current_device()}")
    print(f"✓ Nom du GPU: {torch.cuda.get_device_name(0)}")

    # Propriétés GPU
    props = torch.cuda.get_device_properties(0)
    print(f"✓ VRAM totale: {props.total_memory / 1e9:.2f} GB")
    print(f"✓ VRAM libre: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB utilisés")
    print(f"✓ Compute capability: {props.major}.{props.minor}")

    return True

def test_gpu_computation():
    """Test calcul sur GPU"""
    print_header("2. Test Calcul GPU")

    try:
        import torch

        # Vérifier CUDA
        if not torch.cuda.is_available():
            print("✗ GPU non disponible, skip test calcul")
            return False

        print("Test 1: Multiplication matricielle simple...")
        start = time.time()
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"✓ Multiplication 1000x1000: {elapsed*1000:.2f}ms")

        print("\nTest 2: Opérations en série...")
        start = time.time()
        a = torch.randn(2000, 2000).cuda()
        b = a * 2
        c = b + a
        d = torch.matmul(c, c)
        e = torch.nn.functional.relu(d)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"✓ Opérations complexes: {elapsed*1000:.2f}ms")

        print("\nTest 3: Benchmark throughput...")
        n_iterations = 100
        start = time.time()
        for _ in range(n_iterations):
            x = torch.randn(500, 500).cuda()
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        fps = n_iterations / elapsed
        print(f"✓ Throughput: {fps:.1f} opérations/sec")

        return True

    except Exception as e:
        print(f"✗ Erreur lors du test calcul: {e}")
        return False

def test_memory():
    """Test mémoire GPU"""
    print_header("3. Test Mémoire GPU")

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ GPU non disponible, skip test mémoire")
            return False

        # Mémoire initiale
        torch.cuda.empty_cache()
        mem_initial = torch.cuda.memory_allocated(0) / 1e9
        print(f"Mémoire initiale: {mem_initial:.2f} GB")

        # Allouer de la mémoire
        print("\nAllocation de tenseurs...")
        tensors = []
        for i in range(5):
            size = 1000 * (i + 1)
            t = torch.randn(size, size).cuda()
            tensors.append(t)
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"  Tenseur {i+1} ({size}x{size}): {mem_used:.2f} GB utilisés")

        # Libérer
        print("\nLibération de la mémoire...")
        tensors.clear()
        torch.cuda.empty_cache()
        mem_final = torch.cuda.memory_allocated(0) / 1e9
        print(f"✓ Mémoire finale: {mem_final:.2f} GB")

        return True

    except Exception as e:
        print(f"✗ Erreur lors du test mémoire: {e}")
        return False

def test_mixed_precision():
    """Test mixed precision (FP16)"""
    print_header("4. Test Mixed Precision (FP16)")

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ GPU non disponible, skip test FP16")
            return False

        # Test FP32
        print("Test FP32 (float32)...")
        x_fp32 = torch.randn(2000, 2000).cuda()
        start = time.time()
        for _ in range(50):
            y = torch.matmul(x_fp32, x_fp32)
        torch.cuda.synchronize()
        time_fp32 = time.time() - start
        print(f"  Temps FP32: {time_fp32:.3f}s")

        # Test FP16
        print("\nTest FP16 (float16)...")
        x_fp16 = torch.randn(2000, 2000).half().cuda()
        start = time.time()
        for _ in range(50):
            y = torch.matmul(x_fp16, x_fp16)
        torch.cuda.synchronize()
        time_fp16 = time.time() - start
        print(f"  Temps FP16: {time_fp16:.3f}s")

        speedup = time_fp32 / time_fp16
        print(f"\n✓ Speedup FP16: {speedup:.2f}x plus rapide")

        return True

    except Exception as e:
        print(f"✗ Erreur lors du test FP16: {e}")
        return False

def test_convolution():
    """Test convolution (pour CNN)"""
    print_header("5. Test Convolution (CNN)")

    try:
        import torch
        import torch.nn as nn

        if not torch.cuda.is_available():
            print("✗ GPU non disponible, skip test convolution")
            return False

        # Créer un petit CNN
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()

        # Test avec batch d'images
        print("Test convolution sur batch d'images...")
        batch_sizes = [1, 8, 32, 64]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224).cuda()

            start = time.time()
            for _ in range(10):
                y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            fps = (batch_size * 10) / elapsed
            print(f"  Batch {batch_size:3d}: {fps:6.1f} images/sec")

        print("\n✓ Convolutions fonctionnelles")
        return True

    except Exception as e:
        print(f"✗ Erreur lors du test convolution: {e}")
        return False

def test_rocm_info():
    """Afficher infos ROCm"""
    print_header("6. Informations ROCm")

    import subprocess

    # Test rocminfo
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ rocminfo disponible")

            # Extraire infos GPU
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'Name:' in line and 'gfx' in line:
                    print(f"  {line.strip()}")
                    # Afficher les 5 lignes suivantes
                    for j in range(1, 6):
                        if i+j < len(lines):
                            info_line = lines[i+j].strip()
                            if info_line and not info_line.startswith('*'):
                                print(f"  {info_line}")
        else:
            print("⚠ rocminfo erreur")
    except FileNotFoundError:
        print("⚠ rocminfo non trouvé (ROCm pas installé)")
    except Exception as e:
        print(f"⚠ rocminfo erreur: {e}")

    # Test rocm-smi
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✓ rocm-smi disponible")
            print(result.stdout[:500])  # Afficher les premières lignes
        else:
            print("\n⚠ rocm-smi erreur")
    except FileNotFoundError:
        print("\n⚠ rocm-smi non trouvé")
    except Exception as e:
        print(f"\n⚠ rocm-smi erreur: {e}")

def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("  TEST COMPLET ROCm + PyTorch GPU")
    print("  AMD Radeon RX 7900 XT")
    print("=" * 70)

    results = {}

    # Exécuter tous les tests
    results['pytorch'] = test_pytorch()

    if results['pytorch']:
        results['computation'] = test_gpu_computation()
        results['memory'] = test_memory()
        results['mixed_precision'] = test_mixed_precision()
        results['convolution'] = test_convolution()

    test_rocm_info()

    # Résumé
    print_header("RÉSUMÉ")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTests réussis: {passed_tests}/{total_tests}")
    print("")

    for test_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {test_name}")

    print("\n" + "=" * 70)

    if passed_tests == total_tests:
        print("✓ TOUS LES TESTS RÉUSSIS - GPU FONCTIONNEL !")
        print("=" * 70)
        print("\nVotre système est prêt pour l'entraînement GPU.")
        print("Lancez: python scripts/train.py")
        return 0
    elif results.get('pytorch') and passed_tests > 1:
        print("⚠ TESTS PARTIELLEMENT RÉUSSIS")
        print("=" * 70)
        print("\nLe GPU fonctionne mais certains tests ont échoué.")
        print("Vous pouvez quand même tenter l'entraînement.")
        return 1
    else:
        print("✗ TESTS ÉCHOUÉS - GPU NON FONCTIONNEL")
        print("=" * 70)
        print("\nConsultez le guide: ROCM_WSL2_QUICKSTART.md")
        return 2

if __name__ == "__main__":
    sys.exit(main())
