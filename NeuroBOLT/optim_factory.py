"""
Built upon the LaBraM, BEiT-v2, timm, DeiT, and DINO codebases:

- https://github.com/935963004/LaBraM
- https://github.com/microsoft/unilm/tree/master/beitv2
- https://github.com/rwightman/pytorch-image-models/tree/master/timm
- https://github.com/facebookresearch/deit/
- https://github.com/facebookresearch/dino
"""

import torch
from torch import optim as optim
import json

# Try to import timm optimizers with fallbacks
try:
    from timm.optim.adafactor import Adafactor
except ImportError:
    Adafactor = None

try:
    from timm.optim.adahessian import Adahessian
except ImportError:
    Adahessian = None

try:
    from timm.optim.adamp import AdamP
except ImportError:
    AdamP = None

try:
    from timm.optim.lookahead import Lookahead
except ImportError:
    Lookahead = None

try:
    from timm.optim.nadam import Nadam
except ImportError:
    Nadam = None

try:
    from timm.optim.nvnovograd import NvNovoGrad
except ImportError:
    NvNovoGrad = None

try:
    from timm.optim.radam import RAdam
except ImportError:
    RAdam = None

try:
    from timm.optim.rmsprop_tf import RMSpropTF
except ImportError:
    RMSpropTF = None

try:
    from timm.optim.sgdp import SGDP
except ImportError:
    SGDP = None

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False
    FusedNovoGrad = FusedAdam = FusedLAMB = FusedSGD = None


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        if not (has_apex and torch.cuda.is_available()):
            print("Warning: APEX not available or CUDA not available, falling back to regular optimizers")
            opt_lower = opt_lower.replace('fused', '')

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    
    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam' and Nadam is not None:
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'nadam' and Nadam is None:
        print("Warning: Nadam not available, using Adam instead")
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'radam' and RAdam is not None:
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'radam' and RAdam is None:
        print("Warning: RAdam not available, using Adam instead")
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamp' and AdamP is not None:
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'adamp' and AdamP is None:
        print("Warning: AdamP not available, using AdamW instead")
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'sgdp' and SGDP is not None:
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp' and SGDP is None:
        print("Warning: SGDP not available, using SGD instead")
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor' and Adafactor is not None:
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adafactor' and Adafactor is None:
        print("Warning: Adafactor not available, using AdamW instead")
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adahessian' and Adahessian is not None:
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'adahessian' and Adahessian is None:
        print("Warning: Adahessian not available, using Adam instead")
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf' and RMSpropTF is not None:
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf' and RMSpropTF is None:
        print("Warning: RMSpropTF not available, using RMSprop instead")
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd' and NvNovoGrad is not None:
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd' and NvNovoGrad is None:
        print("Warning: NvNovoGrad not available, using Adam instead")
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'fusedsgd' and FusedSGD is not None:
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum' and FusedSGD is not None:
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam' and FusedAdam is not None:
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw' and FusedAdam is not None:
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb' and FusedLAMB is not None:
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd' and FusedNovoGrad is not None:
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    elif opt_lower.startswith('fused'):
        print(f"Warning: {opt_lower} not available, using AdamW instead")
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        print(f"Warning: Unknown optimizer {opt_lower}, using AdamW instead")
        optimizer = optim.AdamW(parameters, **opt_args)

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead' and Lookahead is not None:
            optimizer = Lookahead(optimizer)
        elif opt_split[0] == 'lookahead' and Lookahead is None:
            print("Warning: Lookahead not available, using base optimizer")

    return optimizer
