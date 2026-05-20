"""
NeuroLens — Smoke Test do Kaggle Kernels

Objetivo: validar o ciclo de execução remota antes de comprometer com a
estrutura completa do projeto.

Verifica:
  1. Imagem do Kaggle subiu com Torch e Tensorflow disponíveis
  2. GPU está visível e acessível
  3. Conseguimos gerar artefatos de saída que voltam pra gente
     via `kaggle kernels output`

Se este script rodar e o `result.json` chegar de volta, o pipeline de
push -> run -> pull está validado.
"""

import json
import platform
import socket
import sys
import time
from pathlib import Path


def collect_environment_info() -> dict:
    """Coleta informações da imagem em que o kernel está rodando."""
    info: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            info["torch_cuda_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
    except ImportError:
        info["torch_version"] = None

    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
        info["tf_gpus_visible"] = [str(d) for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        info["tensorflow_version"] = None

    return info


def run_micro_workload() -> dict:
    """Executa uma multiplicação de matrizes na GPU pra confirmar funcionamento."""
    import torch

    if not torch.cuda.is_available():
        return {"gpu_used": False, "reason": "CUDA não disponível"}

    device = torch.device("cuda")
    size = 4096

    start = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return {
        "gpu_used": True,
        "matrix_size": size,
        "elapsed_seconds": round(elapsed, 4),
        "result_shape": list(c.shape),
        "result_mean": float(c.mean().item()),
    }


def main() -> None:
    print("=" * 60)
    print("NeuroLens — Smoke Test")
    print("=" * 60)

    env_info = collect_environment_info()
    print("\n[1/3] Ambiente:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")

    print("\n[2/3] Workload de GPU:")
    workload_result = run_micro_workload()
    for key, value in workload_result.items():
        print(f"  {key}: {value}")

    output = {
        "status": "ok",
        "environment": env_info,
        "workload": workload_result,
    }

    output_path = Path("/kaggle/working/smoke_test_result.json")
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\n[3/3] Resultado salvo em: {output_path}")
    print("=" * 60)
    print("SMOKE TEST CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()
