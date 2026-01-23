import torch, os, sys
print('torch.__version__ =', torch.__version__)
print('torch.version.hip =', getattr(torch.version, 'hip', None))
print('torch.version.cuda =', getattr(torch.version, 'cuda', None))
print('torch.cuda.is_available() ->', torch.cuda.is_available())
try:
    print('torch.cuda.device_count() ->', torch.cuda.device_count())
except Exception as e:
    print('device_count error ->', type(e).__name__, e)
try:
    import rocm_sdk
    print('rocm_sdk.__version__ ->', rocm_sdk.__version__)
except Exception as e:
    print('rocm_sdk import error ->', type(e).__name__, e)
print('\nEnvironment variables:')
for v in ['ROCM_SDK_PRELOAD_LIBRARIES','ROCM_PATH','ROCM_HOME','ROCM_ROOT','ROCM_VERSION','PATH']:
    print(f'  {v}={os.environ.get(v)}')
print('\nTry small tensor op:')
try:
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2,2, device=d)
    print('  created tensor on', d)
    y = x * 2
    print('  op ok')
except Exception as e:
    print('  tensor op error ->', type(e).__name__, e)
