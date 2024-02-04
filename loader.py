import os
import random
from typing import Any, Dict, List, Tuple, Union
import argparse
from importlib import import_module
#
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
#
from utils.optimizer import Optimizer
from utils.evaluator import TrajPredictionEvaluator


class Loader:
    '''
        Get and return dataset, network, loss_fn, optimizer, evaluator
    '''

    def __init__(self, args, device, is_ddp=False, world_size=1, local_rank=0, verbose=True):
        self.args = args
        self.device = device
        self.is_ddp = is_ddp
        self.world_size = world_size
        self.local_rank = local_rank
        self.resume = False
        self.verbose = verbose

        self.print('[Loader] load adv_cfg from {}'.format(self.args.adv_cfg_path))
        self.adv_cfg = import_module(self.args.adv_cfg_path).AdvCfg()

    def print(self, info):
        if self.verbose:
            print(info)

    def set_resmue(self, model_path):
        self.resume = True
        if not model_path.endswith(".tar"):
            assert False, "Model path error - '{}'".format(model_path)
        self.ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

    def load(self):
        # dataset
        dataset = self.get_dataset()
        # network
        model = self.get_model()
        # loss_fn
        loss_fn = self.get_loss_fn()
        # optimizer
        optimizer = self.get_optimizer(model)
        # evaluator
        evaluator = self.get_evaluator()

        return dataset, model, loss_fn, optimizer, evaluator

    def get_dataset(self):
        data_cfg = self.adv_cfg.get_dataset_cfg()

        ds_file, ds_name = data_cfg['dataset'].split(':')
        self.print('[Loader] load dataset {} from {}'.format(ds_name, ds_file))

        train_dir = self.args.features_dir + 'train/'
        val_dir = self.args.features_dir + 'val/'
        test_dir = self.args.features_dir + 'test/'

        if self.args.mode == 'train' or self.args.mode == 'val':
            train_set = getattr(import_module(ds_file), ds_name)(train_dir,
                                                                 mode='train',
                                                                 obs_len=data_cfg['g_obs_len'],
                                                                 pred_len=data_cfg['g_pred_len'],
                                                                 verbose=self.verbose,
                                                                 aug=self.args.data_aug)
            val_set = getattr(import_module(ds_file), ds_name)(val_dir,
                                                               mode='val',
                                                               obs_len=data_cfg['g_obs_len'],
                                                               pred_len=data_cfg['g_pred_len'],
                                                               verbose=self.verbose,
                                                               aug=False)
            return train_set, val_set
        elif self.args.mode == 'test':
            test_set = getattr(import_module(ds_file), ds_name)(test_dir,
                                                                mode='test',
                                                                obs_len=data_cfg['g_obs_len'],
                                                                pred_len=data_cfg['g_pred_len'],
                                                                verbose=self.verbose)
            return test_set
        else:
            assert False, "Unknown mode"

    def get_model(self):
        net_cfg = self.adv_cfg.get_net_cfg()
        net_file, net_name = net_cfg['network'].split(':')

        self.print('[Loader] load network {} from {}'.format(net_name, net_file))
        model = getattr(import_module(net_file), net_name)(net_cfg, self.device)

        # print network params
        total_num = sum(p.numel() for p in model.parameters())
        self.print('[Loader] network params:')
        self.print('-- total: {}'.format(total_num))
        subnets = list()
        for name, param in model.named_parameters():
            # print(name, param.numel())
            subnets.append(name.split('.')[0])
        subnets = list(set(subnets))
        for subnet in subnets:
            numelem = 0
            for name, param in model.named_parameters():
                if name.startswith(subnet):
                    numelem += param.numel()
            self.print('-- {} {}'.format(subnet, numelem))

        if self.resume:
            model.load_state_dict(self.ckpt["state_dict"])

        if self.is_ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)  # SyncBN
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            model = model.to(self.device)

        return model

    def get_loss_fn(self):
        loss_cfg = self.adv_cfg.get_loss_cfg()
        loss_file, loss_name = loss_cfg['loss_fn'].split(':')

        self.print('[Loader] loading loss {} from {}'.format(loss_name, loss_file))
        loss = getattr(import_module(loss_file), loss_name)(loss_cfg, self.device)
        return loss

    def get_optimizer(self, model):
        opt_cfg = self.adv_cfg.get_opt_cfg()

        if opt_cfg['lr_scale_func'] == 'linear':
            opt_cfg['lr_scale'] = self.world_size
        elif opt_cfg['lr_scale_func'] == 'sqrt':
            opt_cfg['lr_scale'] = np.sqrt(self.world_size)
        else:
            opt_cfg['lr_scale'] = 1.0

        optimizer = Optimizer(model, opt_cfg)

        if self.resume:
            optimizer.load_state_dict(self.ckpt["opt_state"])

        return optimizer

    def get_evaluator(self):
        eval_cfg = self.adv_cfg.get_eval_cfg()
        eval_file, eval_name = eval_cfg['evaluator'].split(':')

        evaluator = getattr(import_module(eval_file), eval_name)(eval_cfg)
        return evaluator

    def network_name(self):
        _, net_name = self.adv_cfg.get_net_cfg()['network'].split(':')
        return net_name
