import torch
import numpy as np
import random
import pynvml
import logging


logger = logging.getLogger('EMOE')


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
        gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    # logger.info("Let's use %d GPUs!" % len(gpu_ids))
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
            # print(p)
    return res

def eva_imp(y_true, y_pred):
    res = (y_pred - y_true) ** 2
    return res

def uni_distill(logits1, logits2):
    prob1 = torch.softmax(logits1, dim=-1)
    prob2 = torch.softmax(logits2, dim=-1)
    mse = torch.mean((prob1 - prob2) ** 2, dim=-1)
    return torch.mean(mse)

def entropy_balance(probs):
    probs = torch.clamp(probs, min=1e-9)
    N = probs.size(1)
    entropy = N * torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(entropy)