import os
import sys


class AdvCfg():
    def __init__(self, is_ddp=False):
        self.g_cfg = dict()
        self.g_cfg['g_num_modes'] = 6
        self.g_cfg['g_obs_len'] = 50
        self.g_cfg['g_pred_len'] = 60

    def get_dataset_cfg(self):
        data_cfg = dict()
        data_cfg['dataset'] = "simpl.av2_dataset:AV2Dataset"

        data_cfg.update(self.g_cfg)  # append global config
        return data_cfg

    def get_net_cfg(self):
        net_cfg = dict()
        net_cfg["network"] = "simpl.simpl:Simpl"
        net_cfg["init_weights"] = False
        net_cfg["in_actor"] = 14
        net_cfg["d_actor"] = 128
        net_cfg["n_fpn_scale"] = 4
        net_cfg["in_lane"] = 16
        net_cfg["d_lane"] = 128

        net_cfg["d_rpe_in"] = 5
        net_cfg["d_rpe"] = 128
        net_cfg["d_embed"] = 128
        net_cfg["n_scene_layer"] = 4
        net_cfg["n_scene_head"] = 8
        net_cfg["dropout"] = 0.1
        net_cfg["update_edge"] = True

        net_cfg["param_out"] = 'bezier'  # bezier/monomial/none
        net_cfg["param_order"] = 7     # 7-th order polynomials

        net_cfg.update(self.g_cfg)  # append global config
        return net_cfg

    def get_loss_cfg(self):
        loss_cfg = dict()
        loss_cfg["loss_fn"] = "simpl.av2_loss_fn:LossFunc"
        loss_cfg["cls_coef"] = 0.1
        loss_cfg["reg_coef"] = 0.9
        loss_cfg["mgn"] = 0.2
        loss_cfg["cls_th"] = 2.0
        loss_cfg["cls_ignore"] = 0.2
        loss_cfg["yaw_loss"] = True

        loss_cfg.update(self.g_cfg)  # append global config
        return loss_cfg

    def get_opt_cfg(self):
        opt_cfg = dict()
        opt_cfg['opt'] = 'adam'
        opt_cfg['weight_decay'] = 0.0
        opt_cfg['lr_scale_func'] = 'none'  # none/sqrt/linear

        # scheduler
        opt_cfg['scheduler'] = 'polyline'

        if opt_cfg['scheduler'] == 'cosine':
            opt_cfg['init_lr'] = 6e-4
            opt_cfg['T_max'] = 50
            opt_cfg['eta_min'] = 1e-5
        elif opt_cfg['scheduler'] == 'cosine_warmup':
            opt_cfg['init_lr'] = 1e-3
            opt_cfg['T_max'] = 50
            opt_cfg['eta_min'] = 1e-4
            opt_cfg['T_warmup'] = 5
        elif opt_cfg['scheduler'] == 'step':
            opt_cfg['init_lr'] = 1e-3
            opt_cfg['step_size'] = 40
            opt_cfg['gamma'] = 0.1
        elif opt_cfg['scheduler'] == 'polyline':
            opt_cfg['init_lr'] = 1e-4
            opt_cfg['milestones'] = [0, 5, 35, 40]
            opt_cfg['values'] = [1e-4, 1e-3, 1e-3, 1e-4]

        opt_cfg.update(self.g_cfg)  # append global config
        return opt_cfg

    def get_eval_cfg(self):
        eval_cfg = dict()
        eval_cfg['evaluator'] = 'utils.evaluator:TrajPredictionEvaluator'
        eval_cfg['data_ver'] = 'av2'
        eval_cfg['miss_thres'] = 2.0

        eval_cfg.update(self.g_cfg)  # append global config
        return eval_cfg
