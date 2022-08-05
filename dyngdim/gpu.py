import torch

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    if devices:
        return devices
    else:
        return [torch.device('cpu')]