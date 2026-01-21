import torch
print('\n' + '='*60)
print('PyTorch Installation Verification')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
    
    # Test GPU computation
    print('\nTesting GPU computation...')
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✅ GPU computation successful!')
else:
    print('❌ No GPU detected - check driver installation')
print('='*60 + '\n')