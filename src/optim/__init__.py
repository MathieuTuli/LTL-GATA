from typing import Dict, Any

import torch

from optim.radam import RAdam


def get_optimizer(net: torch.nn.Module,
                  config: Dict[str, Any]) -> torch.optim.Optimizer:
    # exclude some parameters from optimizer
    param_frozen_list = []  # should be changed into torch.nn.ParameterList()
    param_active_list = []  # should be changed into torch.nn.ParameterList()
    for k, v in net.named_parameters():
        keep_this = True
        for keyword in set(config['fix_parameters_keywords']):
            if keyword in k:
                param_frozen_list.append(v)
                keep_this = False
                break
        if keep_this:
            param_active_list.append(v)

    param_frozen_list = torch.nn.ParameterList(param_frozen_list)
    param_active_list = torch.nn.ParameterList(param_active_list)
    params = [{
        'params': param_frozen_list, 'lr': 0.0},
        {'params': param_active_list, 'lr': config['kwargs']['lr']}]
    optimizer_kwargs = config['kwargs']
    if config['name'] == 'adam':
        optimizer = torch.optim.Adam(params, **optimizer_kwargs)
    elif config['name'] == 'radam':
        optimizer = RAdam(params, **optimizer_kwargs)
    elif config['name'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, **optimizer_kwargs)
    else:
        raise NotImplementedError
    scheduler = None
    scheduler_kwargs = config['scheduler_kwargs']
    if config['scheduler'] == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **scheduler_kwargs)
    return optimizer, scheduler
