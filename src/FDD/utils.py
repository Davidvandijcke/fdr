import torch
from torch.overrides import TorchFunctionMode
import numpy as np

_DEVICE_CONSTRUCTOR = {
    # standard ones
    torch.empty,
    torch.empty_strided,
    torch.empty_quantized,
    torch.empty_like,
    torch.ones,
    torch.arange,
    torch.bartlett_window,
    torch.blackman_window,
    torch.eye,
    torch.fft.fftfreq,
    torch.fft.rfftfreq,
    torch.full,
    torch.fill,
    torch.hamming_window,
    torch.hann_window,
    torch.kaiser_window,
    torch.linspace,
    torch.logspace,
    torch.nested.nested_tensor,
    # torch.normal,
    torch.ones,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.range,
    torch.sparse_coo_tensor,
    torch.sparse_compressed_tensor,
    torch.sparse_csr_tensor,
    torch.sparse_csc_tensor,
    torch.sparse_bsr_tensor,
    torch.sparse_bsc_tensor,
    torch.tril_indices,
    torch.triu_indices,
    torch.vander,
    torch.zeros,
    torch.zeros_like,
    torch.asarray,
    # weird ones
    torch.tensor,
    torch.as_tensor,
}



class DeviceMode(TorchFunctionMode):
    def __init__(self, device):
        self.device = torch.device(device)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _DEVICE_CONSTRUCTOR:
            if kwargs.get('device') is None:
                kwargs['device'] = self.device
            return func(*args, **kwargs)
        return func(*args, **kwargs)
    

def setDevice():
    if torch.cuda.is_available(): # cuda gpus
        device = torch.device("cuda")
        #torch.cuda.set_device(int(gpu_id))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif torch.backends.mps.is_available(): # mac gpus
        device = torch.device("mps")
    elif torch.backends.mkl.is_available(): # intel cpus
        device = torch.device("mkl")
    torch.set_grad_enabled(True)
    return device


def forward_differences(ubar, D : int):

    diffs = []

    for dim in range(int(D)):
        #zeros_shape = torch.jit.annotate(List[int], [])
        zeros_shape = list(ubar.shape)
        zeros_shape[dim] = 1
        # for j in range(ubar.dim()): 
        #     if j == dim:
        #         zeros_shape.append(1)
        #     else:
        #         zeros_shape.append(ubar.shape[j])
        zeros = np.zeros(zeros_shape)
        diff = np.concatenate((np.diff(ubar, axis=dim), zeros), axis=dim)
        diffs.append(diff)

    # Stack the results along a new dimension (first dimension)
    u_star = np.stack(diffs, axis=0)

    return u_star

def isosurface(u, level, grid_y):

    mask = (u[...,:-1] > 0.5) & (u[...,1:] <= 0.5)
    # Find the indices of the first True value along the last dimension, and set all the following ones to False
    mask[..., 1:] = (mask[..., 1:]) & (mask.cumsum(-1)[...,:-1] < 1)

    uk0 = u[...,:-1][mask]
    uk1 = u[...,1:][mask]
    
    # get the indices of the last dimension where mask is True
    k = np.where(mask == True)[-1] + 1
    
    h_img = interpolate(k, uk0, uk1, level).reshape(grid_y.shape[:-1])
    
    return h_img

def interpolate(k, uk0, uk1, l):
    return (k + (0.5 - uk0) / (uk1 - uk0)) / l