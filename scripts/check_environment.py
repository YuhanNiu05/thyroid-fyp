"""Read-only environment snapshot and tiny CUDA operation (no training)."""
import datetime
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
          'disk': {str(p): dict(zip(['total_bytes', 'used_bytes', 'free_bytes'], shutil.disk_usage(p))) for p in [root, Path('/tmp')]}}
for command in [['nvidia-smi'], ['/usr/local/cuda/bin/nvcc', '--version']]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result[' '.join(command)] = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except Exception as exc:
        result[' '.join(command)] = {'error': str(exc)}
for name in ['torch', 'PIL', 'numpy', 'transformers', 'datasets', 'cv2']:
    try:
        module = importlib.import_module(name)
        result[name] = {'version': module.__version__, 'path': module.__file__}
        if name == 'torch':
            result[name].update(cuda_runtime=module.version.cuda, cuda_available=module.cuda.is_available(), device_count=module.cuda.device_count())
            if module.cuda.is_available():
                result[name]['gpu'] = module.cuda.get_device_name(0)
                result[name]['memory_bytes'] = module.cuda.get_device_properties(0).total_memory
                result[name]['cuda_smoke_test'] = (module.ones(3, device='cuda') + 1).tolist()
    except Exception as exc:
        result[name] = {'error': str(exc)}
(root / 'evidence/environment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
