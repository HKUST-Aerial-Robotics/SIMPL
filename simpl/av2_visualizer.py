import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.vis_utils import AV2MapVisualizer


_ESTIMATED_VEHICLE_LENGTH_M = 5.0
_ESTIMATED_VEHICLE_WIDTH_M = 2.0
_ESTIMATED_CYCLIST_LENGTH_M = 2.0
_ESTIMATED_CYCLIST_WIDTH_M = 0.7
_ESTIMATED_PEDESTRIAN_LENGTH_M = 0.3
_ESTIMATED_PEDESTRIAN_WIDTH_M = 0.5
_ESTIMATED_BUS_LENGTH_M = 7.0
_ESTIMATED_BUS_WIDTH_M = 2.1


class Visualizer():
    def __init__(self):
        self.map_vis = AV2MapVisualizer()

    def draw_once(self, post_out, data, eval_out, show_map=False, test_mode=False, split='val'):
        batch_size = len(data['SEQ_ID'])

        seq_id = data['SEQ_ID'][0]
        city_name = data['CITY_NAME'][0]
        orig = data['ORIG'][0]
        rot = data['ROT'][0]
        trajs_obs = data['TRAJS'][0]['TRAJS_POS_OBS']
        trajs_fut = data['TRAJS'][0]['TRAJS_POS_FUT']
        pads_obs = data['TRAJS'][0]['PAD_OBS']
        pads_fut = data['TRAJS'][0]['PAD_FUT']
        trajs_ctrs = data['TRAJS'][0]['TRAJS_CTRS']
        trajs_vecs = data['TRAJS'][0]['TRAJS_VECS']
        #
        trajs_type = data['TRAJS'][0]["TRAJS_TYPE"]
        trajs_tid = data['TRAJS'][0]["TRAJS_TID"]
        trajs_cat = data['TRAJS'][0]["TRAJS_CAT"]
        #
        lane_graph = data['LANE_GRAPH'][0]

        res_cls, res_reg, res_aux = post_out['out_raw']

        print("[Vis] seq_id: {}, city_name: {}".format(seq_id, city_name))
        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        ax.set_title('{}-{}'.format(seq_id, city_name))

        if show_map:
            self.map_vis.show_map_clean(ax, split, seq_id)
            pass
        else:
            rot = torch.eye(2)
            orig = torch.zeros(2)

        # trajs hist
        for i, (traj_obs, pad_obs, ctr, vec) in enumerate(zip(trajs_obs, pads_obs, trajs_ctrs, trajs_vecs)):
            traj_cat = trajs_cat[i]
            zorder = 10
            if traj_cat == 'focal':
                clr = 'r'
                zorder = 20
            elif traj_cat == 'av':
                clr = 'cornflowerblue'
            elif traj_cat == 'score':
                clr = 'royalblue'
            else:
                clr = 'grey'

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            if not traj_cat in ['unscore', 'frag']:
                traj_obs = torch.matmul(traj_obs, act_rot.T) + ctr
                traj_obs = torch.matmul(traj_obs, rot.T) + orig
                ax.plot(traj_obs[:, 0], traj_obs[:, 1], marker='.', alpha=0.25, color=clr, zorder=zorder)

            # attrs
            traj_type = np.where(trajs_type[i][len(traj_obs) - 1])[0][0]
            if traj_type == 0:
                bbox_l = _ESTIMATED_VEHICLE_LENGTH_M
                bbox_w = _ESTIMATED_VEHICLE_WIDTH_M
            elif traj_type == 1:
                bbox_l = _ESTIMATED_PEDESTRIAN_LENGTH_M
                bbox_w = _ESTIMATED_PEDESTRIAN_WIDTH_M
            elif traj_type == 2 or traj_type == 3:
                bbox_l = _ESTIMATED_CYCLIST_LENGTH_M
                bbox_w = _ESTIMATED_CYCLIST_WIDTH_M
            elif traj_type == 4:
                bbox_l = _ESTIMATED_BUS_LENGTH_M
                bbox_w = _ESTIMATED_BUS_WIDTH_M
            elif traj_type == 5:
                bbox_l = 1.0  # unknown
                bbox_w = 1.0
            else:
                bbox_l = 0.5  # static
                bbox_w = 0.5
            bbox = torch.Tensor([[-bbox_l / 2, -bbox_w / 2],
                                 [-bbox_l / 2, bbox_w / 2],
                                 [bbox_l / 2, bbox_w / 2],
                                 [bbox_l / 2, -bbox_w / 2]])
            bbox = torch.matmul(bbox, act_rot.T) + ctr
            bbox = torch.matmul(bbox, rot.T) + orig
            ax.fill(bbox[:, 0], bbox[:, 1], color=clr, alpha=0.5, zorder=zorder)

            ctr_vis = torch.matmul(ctr, rot.T) + orig
            vec_vis = torch.matmul(vec, rot.T)
            ax.arrow(ctr_vis[0], ctr_vis[1], vec_vis[0], vec_vis[1], alpha=0.5, color=clr, width=0.05, zorder=zorder)

        # if not test mode, vis GT trajectories
        if not test_mode:
            for i, (traj_fut, pad_fut, ctr, vec) in enumerate(zip(trajs_fut, pads_fut, trajs_ctrs, trajs_vecs)):
                traj_cat = trajs_cat[i]
                zorder = 10
                if traj_cat == 'focal':
                    clr = 'deeppink'
                    zorder = 20
                elif traj_cat == 'av':
                    clr = 'deepskyblue'
                elif traj_cat == 'score':
                    clr = 'deepskyblue'
                else:
                    clr = 'grey'

                if traj_cat in ['unscore', 'frag']:
                    continue

                theta = np.arctan2(vec[1], vec[0])
                act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

                traj_fut = torch.matmul(traj_fut, act_rot.T) + ctr
                traj_fut = torch.matmul(traj_fut, rot.T) + orig
                ax.plot(traj_fut[:, 0], traj_fut[:, 1], alpha=0.5, color=clr, zorder=zorder)

                # mk = '*' if torch.sum(pad_fut) == 60 else 's'
                mk = '*'
                ax.plot(traj_fut[-1, 0], traj_fut[-1, 1], marker=mk, alpha=0.5, color=clr, zorder=zorder, markersize=10)

        # traj pred all
        print('res_reg: ', [x.shape for x in res_reg])
        res_reg = res_reg[0].cpu().detach()
        res_cls = res_cls[0].cpu().detach()
        for i, (trajs, probs, ctr, vec) in enumerate(zip(res_reg, res_cls, trajs_ctrs, trajs_vecs)):
            traj_cat = trajs_cat[i]
            zorder = 10
            if traj_cat == 'focal':
                clr = 'r'
                zorder = 20
            elif traj_cat == 'av':
                clr = 'cornflowerblue'
            elif traj_cat == 'score':
                clr = 'royalblue'
            else:
                clr = 'grey'

            if traj_cat in ['unscore', 'frag']:
                continue

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            for traj, prob in zip(trajs, probs):
                if prob < 0.05 and (not i in [0]):
                    continue

                pos = torch.matmul(traj[..., 0:2], act_rot.T) + ctr
                pos = torch.matmul(pos[..., 0:2], rot.T) + orig

                ax.plot(pos[:, 0], pos[:, 1], alpha=0.3, color=clr, zorder=zorder, linestyle='--')
                ax.arrow(pos[-2, 0],
                         pos[-2, 1],
                         (pos[-1, 0] - pos[-2, 0]),
                         (pos[-1, 1] - pos[-2, 1]),
                         edgecolor=None,
                         color=clr,
                         alpha=0.3,
                         width=0.2,
                         zorder=zorder)
                ax.text(pos[-1, 0], pos[-1, 1], '{:.2f}'.format(prob),
                        color=clr, zorder=zorder, alpha=0.6)  # rotation=30

        # # lane graph
        # node_ctrs = lane_graph['node_ctrs']  # [196, 10, 2]
        # node_vecs = lane_graph['node_vecs']  # [196, 10, 2]
        # lane_ctrs = lane_graph['lane_ctrs']  # [196, 2]
        # lane_vecs = lane_graph['lane_vecs']  # [196, 2]

        # for ctrs_tmp, vecs_tmp, anch_pos, anch_vec in zip(node_ctrs, node_vecs, lane_ctrs, lane_vecs):
        #     anch_rot = torch.Tensor([[anch_vec[0], -anch_vec[1]],
        #                              [anch_vec[1], anch_vec[0]]])
        #     ctrs_tmp = torch.matmul(ctrs_tmp, anch_rot.T) + anch_pos
        #     ctrs_tmp = torch.matmul(ctrs_tmp, rot.T) + orig
        #     ax.plot(ctrs_tmp[:, 0], ctrs_tmp[:, 1], alpha=0.1, color='grey', linestyle='--')

        plt.tight_layout()
        plt.show()
