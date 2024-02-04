import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.vis_utils import ArgoMapVisualizer


class Visualizer():
    def __init__(self):
        self.map_vis = ArgoMapVisualizer()

    def draw_once(self, post_out, data, eval_out, show_map=False, test_mode=False, split='val'):
        batch_size = len(data['SEQ_ID'])

        seq_id = data['SEQ_ID'][0]
        city_name = data['CITY_NAME'][0]
        orig = data['ORIG'][0]
        rot = data['ROT'][0]
        trajs_obs = data['TRAJS_OBS'][0]
        trajs_fut = data['TRAJS_FUT'][0]
        pads_obs = data['PAD_OBS'][0]
        pads_fut = data['PAD_FUT'][0]
        trajs_ctrs = data['TRAJS_CTRS'][0]
        trajs_vecs = data['TRAJS_VECS'][0]
        lane_graph = data['LANE_GRAPH'][0]

        res_cls = post_out['out_raw'][0]
        res_reg = post_out['out_raw'][1]

        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        ax.set_title('{}-{}'.format(seq_id, city_name))

        if show_map:
            self.map_vis.show_surrounding_elements(ax, city_name, orig)
        else:
            rot = torch.eye(2)
            orig = torch.zeros(2)

        # trajs
        for i, (traj_obs, pad_obs, ctr, vec) in enumerate(zip(trajs_obs, pads_obs, trajs_ctrs, trajs_vecs)):
            zorder = 10
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'cornflowerblue'
            else:
                clr = 'royalblue'

            # if torch.sum(pad_obs) < 15:
            if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                clr = 'grey'

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            traj_obs = torch.matmul(traj_obs, act_rot.T) + ctr
            traj_obs = torch.matmul(traj_obs, rot.T) + orig
            ax.plot(traj_obs[:, 0], traj_obs[:, 1], marker='.', alpha=0.5, color=clr, zorder=zorder)
            ax.plot(traj_obs[-1, 0], traj_obs[-1, 1], marker='o', alpha=0.5, color=clr, zorder=zorder, markersize=10)

        if not test_mode:
            # if not test mode, vis GT trajectories
            for i, (traj_fut, pad_fut, ctr, vec) in enumerate(zip(trajs_fut, pads_fut, trajs_ctrs, trajs_vecs)):
                zorder = 10
                if i == 0:
                    clr = 'deeppink'
                    zorder = 20
                elif i == 1:
                    clr = 'deepskyblue'
                else:
                    clr = 'deepskyblue'

                if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                    continue

                theta = np.arctan2(vec[1], vec[0])
                act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

                traj_fut = torch.matmul(traj_fut, act_rot.T) + ctr
                traj_fut = torch.matmul(traj_fut, rot.T) + orig
                ax.plot(traj_fut[:, 0], traj_fut[:, 1], alpha=0.5, color=clr, zorder=zorder)

                mk = '*' if torch.sum(pad_fut) == 30 else '*'
                ax.plot(traj_fut[-1, 0], traj_fut[-1, 1], marker=mk, alpha=0.5, color=clr, zorder=zorder, markersize=12)

        # traj pred all
        # print('res_reg: ', [x.shape for x in res_reg])
        res_reg = res_reg[0].cpu().detach()
        res_cls = res_cls[0].cpu().detach()
        for i, (trajs, probs, ctr, vec) in enumerate(zip(res_reg, res_cls, trajs_ctrs, trajs_vecs)):
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'cornflowerblue'
            else:
                clr = 'royalblue'

            if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                continue

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            for traj, prob in zip(trajs, probs):
                if prob < 0.05 and (not i in [0, 1]):
                    continue
                traj = torch.matmul(traj, act_rot.T) + ctr
                traj = torch.matmul(traj, rot.T) + orig
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, color=clr, zorder=zorder, linestyle='--')
                # ax.plot(traj[-1, 0], traj[-1, 1], alpha=0.3, marker='o', color=clr, zorder=zorder, markersize=12)

                ax.arrow(traj[-2, 0],
                         traj[-2, 1],
                         (traj[-1, 0] - traj[-2, 0]),
                         (traj[-1, 1] - traj[-2, 1]),
                         edgecolor=None,
                         color=clr,
                         alpha=0.3,
                         width=0.2,
                         zorder=zorder)

        # lane graph
        node_ctrs = lane_graph['node_ctrs']  # [196, 10, 2]
        node_vecs = lane_graph['node_vecs']  # [196, 10, 2]
        lane_ctrs = lane_graph['lane_ctrs']  # [196, 2]
        lane_vecs = lane_graph['lane_vecs']  # [196, 2]

        for ctrs_tmp, vecs_tmp, anch_pos, anch_vec in zip(node_ctrs, node_vecs, lane_ctrs, lane_vecs):
            anch_rot = torch.Tensor([[anch_vec[0], -anch_vec[1]],
                                     [anch_vec[1], anch_vec[0]]])
            ctrs_tmp = torch.matmul(ctrs_tmp, anch_rot.T) + anch_pos
            ctrs_tmp = torch.matmul(ctrs_tmp, rot.T) + orig
            ax.plot(ctrs_tmp[:, 0], ctrs_tmp[:, 1], alpha=0.1, linestyle='dotted', color='grey')

        plt.tight_layout()
        plt.show()
