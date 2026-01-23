import os
# Ensure ROCm / MIOpen env vars and PATH are set before torch is imported
os.environ.setdefault('MIOPEN_LOG_LEVEL', '6')
os.environ.setdefault('MIOPEN_FIND_MODE', '1')
# Build a safe ROCM_SDK_PRELOAD_LIBRARIES value from available rocm_sdk entries
desired_libs = ['amdhip64','hipblas','hipblaslt','hipfft','hiprand','hipsparse','hipsolver','miopen','rocblas','rocm-openblas']
available_libs = []
try:
    # Query rocm_sdk dist info to know which shortnames are present
    import importlib
    d = importlib.import_module('rocm_sdk._dist_info')
    for name in desired_libs:
        if name in getattr(d, 'ALL_LIBRARIES', {}):
            available_libs.append(name)
except Exception:
    # If we can't query rocm_sdk, fall back to a conservative list
    available_libs = [l for l in desired_libs if l != 'rocblas']

if available_libs:
    os.environ['ROCM_SDK_PRELOAD_LIBRARIES'] = ','.join(available_libs)
# Prepend ROCm DLL folders from AMD-managed Python so rocm_sdk can find them
rocm_core_bin = r"C:\Users\paron\AppData\Local\Programs\Python\Python312\Lib\site-packages\_rocm_sdk_core\bin"
rocm_libs_bin = r"C:\Users\paron\AppData\Local\Programs\Python\Python312\Lib\site-packages\_rocm_sdk_libraries_custom\bin"
os.environ['PATH'] = rocm_core_bin + os.pathsep + rocm_libs_bin + os.pathsep + os.environ.get('PATH', '')

import torch, traceback
print('torch.__version__ =', torch.__version__)
print('torch.version.hip =', getattr(torch.version, 'hip', None))
print('torch.cuda.is_available() ->', torch.cuda.is_available())
try:
    print('device_count ->', torch.cuda.device_count())
except Exception as e:
    print('device_count error ->', e)
print('\nEnvironment:')
for k in ['MIOPEN_LOG_LEVEL','MIOPEN_FIND_MODE','ROCM_SDK_PRELOAD_LIBRARIES','ROCM_PATH','ROCM_HOME','ROCM_ROOT','ROCM_VERSION']:
    print(f'  {k}={os.environ.get(k)}')
print('\nTry conv forward+backward on GPU:')
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(8, 3, 64, 64, device=device, requires_grad=True)
    w = torch.randn(16, 3, 3, 3, device=device, requires_grad=True)
    y = torch.nn.functional.conv2d(x, w, padding=1)
    loss = y.mean()
    loss.backward()
    print('Conv/backward OK')
except Exception as e:
    print('Exception during conv/backward:')
    traceback.print_exc()
    # try a simple tensor op to ensure GPU basic ops work
    try:
        t = torch.randn(2,2, device=device)
        print('Basic tensor op OK on', device)
    except Exception as e2:
        print('Basic tensor op failed:')
        traceback.print_exc()
