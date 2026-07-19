import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import warnings
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf


from ssc_ia import LitModule, build_data_loaders, pre_build_callbacks, SetSeed


import torch
import torch.nn.functional as F
import inspect

_orig_batch_norm = F.batch_norm

def my_batch_norm(input, *args, **kwargs):
    # Check if total elements per channel equals 1 (triggers the crash)
    num_elements_per_channel = input.numel() / input.shape[1]
    
    if num_elements_per_channel == 1:
        sig = inspect.signature(_orig_batch_norm)
        bound = sig.bind_partial(input, *args, **kwargs)
        
        # 1. Force evaluation mode behavior
        bound.arguments['training'] = False
        
        # 2. Check if running statistics arrays are missing/None
        rm = bound.arguments.get('running_mean', None)
        rv = bound.arguments.get('running_var', None)
        
        # 3. Dynamically inject dummy statistics if they don't exist
        num_features = input.shape[1]
        if rm is None:
            bound.arguments['running_mean'] = torch.zeros(num_features, device=input.device, dtype=input.dtype)
        if rv is None:
            bound.arguments['running_var'] = torch.ones(num_features, device=input.device, dtype=input.dtype)
            
        return _orig_batch_norm(*bound.args, **bound.kwargs)
        
    return _orig_batch_norm(input, *args, **kwargs)

# Overwrite PyTorch's internal function globally for this execution context
F.batch_norm = my_batch_norm

@hydra.main(config_path='../configs', config_name='config_360', version_base=None)  # my_config
def main(cfg: DictConfig):
    # torch.backends.cudnn.benchmark = False 
    # torch.backends.cudnn.deterministic = True
    
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    
    SetSeed(seed=42)
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, **meta_info)
    trainer = L.Trainer(**cfg.trainer, **callbacks)
    ckpt_path = None if cfg.resume.ckpt_path == 'None' else cfg.resume.ckpt_path

    if cfg.get('validate', False):
        trainer.validate(model, dls[1], ckpt_path=ckpt_path)
    else:
        trainer.fit(model, *dls[:2], ckpt_path=ckpt_path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, message=".*__floordiv__ is deprecated.*")
    main()
