import torch
from typing import Tuple, List, Dict, Union
from pprint import pformat
import random
import numpy


def compile_model(m, fast):
    if fast == 0:
        return m
    return (
        torch.compile(
            m,
            mode={
                1: "reduce-overhead",
                2: "max-autotune",
                3: "default",
            }[fast],
        )
        if hasattr(torch, "compile")
        else m
    )

def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)               # Python 内置随机数
    numpy.random.seed(seed)            # NumPy 随机数
    torch.manual_seed(seed)         # CPU 上的 Torch 随机数
    torch.cuda.manual_seed(seed)    # GPU 上的 Torch 随机数
    torch.cuda.manual_seed_all(seed)  # 多个 GPU 的情况

def filter_params(model, ndim_dict, nowd_keys=()) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)
        
        if ndim_dict.get(name, 0) == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        else:
            cur_wd_sc, group_name = 1., 'D'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')
    
    for rk in range(torch.distributed.get_world_size()):
        torch.distributed.barrier()
        if torch.distributed.get_rank() == rk:
            print(f'[get_param_groups][rank{torch.distributed.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True)
    print('')
    
    assert len(names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    del ndim_dict
    return names, paras, list(para_groups.values())

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = numpy.array([1., ])
    elif(filt_size == 2):
        a = numpy.array([1., 1.])
    elif(filt_size == 3):
        a = numpy.array([1., 2., 1.])
    elif(filt_size == 4):
        a = numpy.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = numpy.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = numpy.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = numpy.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :]) # type: ignore
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = torch.nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = torch.nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = torch.nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer # type: ignore
