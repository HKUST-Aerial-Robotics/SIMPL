import os
import warnings
from typing import Any, Dict, List, Tuple
#
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolylineLR(_LRScheduler):
    def __init__(self, optimizer, milestones, values, last_epoch=-1, verbose=False):

        assert len(milestones) == len(values), '[PolylineLR] length must be same'
        assert all(x >= 0 for x in milestones), '[PolylineLR] milestones must be positive'
        assert all(x >= 0 for x in values), '[PolylineLR] values must be positive'
        assert milestones[0] == 0, '[PolylineLR] milestones must start from 0'
        assert milestones == sorted(milestones), '[PolylineLR] milestones must be in ascending order'

        self.milestones = milestones
        self.values = values
        self.n_intervals = len(self.milestones) - 1
        super(PolylineLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        lr = self._get_value(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]

    def _get_value(self, epoch):
        assert epoch >= 0
        for i in range(self.n_intervals):
            e_lb = self.milestones[i]
            e_ub = self.milestones[i + 1]
            if epoch < e_lb or epoch >= e_ub:
                continue  # not in this interval
            v_lb = self.values[i]
            v_ub = self.values[i + 1]
            v_e = (epoch - e_lb) / (e_ub - e_lb) * (v_ub - v_lb) + v_lb
            return v_e
        return self.values[-1]


class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min, T_warmup, last_epoch=-1, verbose=False):
        self.T_max = T_max - T_warmup
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.init_lr = optimizer.param_groups[0]['lr']
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.T_warmup:
            lr = (self.last_epoch / self.T_warmup) * (self.init_lr - self.eta_min) + self.eta_min
            return [lr for group in self.optimizer.param_groups]

        net_last_epoch = self.last_epoch - self.T_warmup
        if self._step_count == 1 and net_last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(net_last_epoch * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (net_last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * net_last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (net_last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class Optimizer(object):
    def __init__(self, net, config):
        self.config = config

        # scale factor for DDP
        lr_scale = config['lr_scale'] if 'lr_scale' in config else 1.0

        # optimizer
        opt = config["opt"]
        if opt == "sgd":
            self.opt = torch.optim.SGD(net.parameters(),
                                       lr=config['init_lr']*lr_scale,
                                       weight_decay=config['weight_decay'])
        elif opt == "adam":
            self.opt = torch.optim.Adam(net.parameters(),
                                        lr=config['init_lr']*lr_scale,
                                        weight_decay=config['weight_decay'])
        elif opt == "adamw":
            self.opt = torch.optim.AdamW(net.parameters(),
                                         lr=config['init_lr']*lr_scale,
                                         weight_decay=config['weight_decay'])
        else:
            assert False, 'unknown opt type, should be sgd/adam'

        # scheduler
        if config["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt,
                                                                        T_max=config['T_max'],
                                                                        eta_min=config['eta_min']*lr_scale)
        elif config["scheduler"] == "cosine_warmup":
            self.scheduler = CosineAnnealingWithWarmupLR(optimizer=self.opt,
                                                         T_max=config['T_max'],
                                                         eta_min=config['eta_min']*lr_scale,
                                                         T_warmup=config['T_warmup'])
        elif config["scheduler"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.opt,
                                                             step_size=config['step_size'],
                                                             gamma=config['gamma'])
        elif config["scheduler"] == "polyline":
            values = [x * lr_scale for x in config['values']]
            self.scheduler = PolylineLR(optimizer=self.opt,
                                        milestones=config['milestones'],
                                        values=values)
        elif config["scheduler"] == "none":
            self.scheduler = None
        else:
            assert False, 'unknown scheduler type, should be cosine/step/none'

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()
        cur_lr = self.opt.param_groups[0]['lr']
        return cur_lr

    def step_scheduler(self):
        if self.scheduler is None:
            return self.opt.param_groups[0]['lr']
        else:
            self.scheduler.step()
            return self.scheduler.get_last_lr()[0]

    def current_lr(self):
        return self.opt.param_groups[0]['lr']

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)

    def print(self):
        print('\noptimizer config:', self.config)
