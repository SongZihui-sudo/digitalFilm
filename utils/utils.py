import torch
from typing import Tuple, List, Dict, Union, Optional
from pprint import pformat
import random
import numpy
import os
import numpy as np
import requests
import io
import base64
from PIL import Image


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

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    arr = tensor.numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def decode_base64_image(image_base64: str) -> Image.Image:
    if image_base64.startswith("data:image") and "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def load_image_from_url(url: str, timeout: int = 15) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def resize_to_multiple_of_16(img: Image.Image, max_size: Optional[int] = 1536) -> Image.Image:
    w, h = img.size
    if max_size is not None:
        scale = min(max_size / max(w, h), 1.0)
        w = int(w * scale)
        h = int(h * scale)
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    return img.resize((w, h), Image.LANCZOS)
