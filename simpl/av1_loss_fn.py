from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long


class LossFunc(nn.Module):
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, data):
        # print('TRAJS_FUT: ', len(data["TRAJS_FUT"]), data["TRAJS_FUT"][0].shape)
        # print('PAD_FUT: ', len(data["PAD_FUT"]), data["PAD_FUT"][0].shape)
        # print('out: ', out[1][0].shape, out[0][0].shape)
        loss_out = self.pred_loss(out,
                                  gpu(data["TRAJS_FUT"], self.device),
                                  to_long(gpu(data["PAD_FUT"], self.device)))
        loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"]
        return loss_out

    def pred_loss(self, out: Dict[str, List[torch.Tensor]], gt_preds: List[torch.Tensor], pad_flags: List[torch.Tensor]):
        cls = out[0]
        reg = out[1]
        # cls = torch.cat([x[0:2] for x in cls], 0)
        # reg = torch.cat([x[0:2] for x in reg], 0)
        # gt_preds = torch.cat([x[0:2] for x in gt_preds], 0)
        # has_preds = torch.cat([x[0:2] for x in pad_flags], 0).bool()
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in pad_flags], 0).bool()

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = 30
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        _reg = reg[..., 0:2].clone()  # for WTA strategy, in case of (5-dim) prob output

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        num_cls = mask.sum().item()
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss

        reg = reg[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss

        return loss_out

    def print(self):
        print('\nloss_fn config:', self.config)
        print('loss_fn device:', self.device)
