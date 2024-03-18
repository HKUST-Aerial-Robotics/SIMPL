import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
import torch
from torch.utils.data import Dataset
#
from utils.utils import from_numpy


class AV2Dataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 50,
                 pred_len: int = 60,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("[Dataset] preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                sequences = sorted(sequences)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            sequences = sorted(sequences)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        df = pd.read_pickle(self.dataset_files[idx])
        '''
            "SEQ_ID", "CITY_NAME", "ORIG", "ROT",
            "TRAJS", "LANE_GRAPH"
        '''

        data = self.data_augmentation(df)

        seq_id = data['SEQ_ID']
        city_name = data['CITY_NAME']
        orig = data['ORIG']
        rot = data['ROT']

        _trajs = data['TRAJS']

        # split obs and fut
        trajs_pos_obs = _trajs['trajs_pos'][:, :self.obs_len]
        trajs_ang_obs = _trajs['trajs_ang'][:, :self.obs_len]
        trajs_vel_obs = _trajs['trajs_vel'][:, :self.obs_len]
        pad_obs = _trajs['has_flags'][:, :self.obs_len]
        trajs_type = _trajs['trajs_type'][:, :self.obs_len]
        trajs_pos_fut = _trajs['trajs_pos'][:, self.obs_len:]
        trajs_ang_fut = _trajs['trajs_ang'][:, self.obs_len:]
        trajs_vel_fut = _trajs['trajs_vel'][:, self.obs_len:]
        pad_fut = _trajs['has_flags'][:, self.obs_len:]
        #
        trajs_ctrs = _trajs['trajs_ctrs']
        trajs_vecs = _trajs['trajs_vecs']

        trajs = dict()
        # observation
        trajs["TRAJS_POS_OBS"] = trajs_pos_obs
        trajs["TRAJS_ANG_OBS"] = np.stack([np.cos(trajs_ang_obs), np.sin(trajs_ang_obs)], axis=-1)
        trajs["TRAJS_VEL_OBS"] = trajs_vel_obs
        trajs["TRAJS_TYPE"] = trajs_type
        trajs["PAD_OBS"] = pad_obs
        # ground truth
        trajs["TRAJS_POS_FUT"] = trajs_pos_fut
        trajs["TRAJS_ANG_FUT"] = np.stack([np.cos(trajs_ang_fut), np.sin(trajs_ang_fut)], axis=-1)
        trajs["TRAJS_VEL_FUT"] = trajs_vel_fut
        trajs["PAD_FUT"] = pad_fut
        # anchor ctrs & vecs
        trajs["TRAJS_CTRS"] = trajs_ctrs
        trajs["TRAJS_VECS"] = trajs_vecs
        # track id & category
        trajs["TRAJS_TID"] = _trajs['trajs_tid']  # List[str]
        trajs["TRAJS_CAT"] = _trajs['trajs_cat']  # List[str]
        # ~ training mask
        # disable unknown and static unless it is focal, av, or score
        # train_mask = []
        # for typ, cat in zip(trajs_type[:, self.obs_len - 1], _trajs['trajs_cat']):
        #     if (np.where(typ)[0][0] in [5, 6]) and (cat not in ['focal', 'av', 'score']):
        #         train_mask.append(False)
        #     else:
        #         train_mask.append(True)
        # trajs["TRAIN_MASK"] = np.array(train_mask, dtype=bool)
        trajs["TRAIN_MASK"] = np.ones(len(trajs_ctrs), dtype=bool)  # * train all mot
        # print('trajs_ctrs: ', trajs_ctrs.shape)
        # print('TRAIN_MASK: ', trajs["TRAIN_MASK"].shape, trajs["TRAIN_MASK"])

        # ~ yaw loss mask (for MOT with non-holonomic constraints)
        yaw_loss_mask = np.array([np.where(x)[0][0] in [0, 2, 3, 4] for x in trajs_type[:, -1]], dtype=bool)
        trajs["YAW_LOSS_MASK"] = yaw_loss_mask
        # print('yaw_loss_mask: ', trajs["YAW_LOSS_MASK"].shape, trajs["YAW_LOSS_MASK"])

        graph = data['LANE_GRAPH']
        # for k, v in graph.items():
        #     print(k, type(v), v.shape if type(v) == np.ndarray else [])

        lane_ctrs = graph['lane_ctrs']
        lane_vecs = graph['lane_vecs']

        # ~ calc rpe
        rpes = dict()
        scene_ctrs = torch.cat([torch.from_numpy(trajs_ctrs), torch.from_numpy(lane_ctrs)], dim=0)
        scene_vecs = torch.cat([torch.from_numpy(trajs_vecs), torch.from_numpy(lane_vecs)], dim=0)
        rpes['scene'], rpes['scene_mask'] = self._get_rpe(scene_ctrs, scene_vecs)

        data = {}
        data['SEQ_ID'] = seq_id
        data['CITY_NAME'] = city_name
        data['ORIG'] = orig
        data['ROT'] = rot
        data['TRAJS'] = trajs
        data['LANE_GRAPH'] = graph
        data['RPE'] = rpes

        return data

    def _get_cos(self, v1, v2):
        ''' input: [M, N, 2], [M, N, 2]
            output: [M, N]
            cos(<a,b>) = (a dot b) / |a||b|
        '''
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos_dang

    def _get_sin(self, v1, v2):
        ''' input: [M, N, 2], [M, N, 2]
            output: [M, N]
            sin(<a,b>) = (a x b) / |a||b|
        '''
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin_dang

    def _get_rpe(self, ctrs, vecs, radius=100.0):
        # distance encoding
        d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
        if False:
            mask = d_pos >= radius
            d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
            pos_rpe = []
            for l_pos in range(10):
                pos_rpe.append(torch.sin(2**l_pos * math.pi * d_pos))
                pos_rpe.append(torch.cos(2**l_pos * math.pi * d_pos))
            # print('pos rpe: ', [x.shape for x in pos_rpe])
            pos_rpe = torch.stack(pos_rpe)
            # print('pos_rpe: ', pos_rpe.shape)
        else:
            mask = None
            d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
            pos_rpe = d_pos.unsqueeze(0)
            # print('pos_rpe: ', pos_rpe.shape)

        # angle diff
        cos_a1 = self._get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
        sin_a1 = self._get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
        # print('cos_a1: ', cos_a1.shape, 'sin_a1: ', sin_a1.shape)

        v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
        cos_a2 = self._get_cos(vecs.unsqueeze(0), v_pos)
        sin_a2 = self._get_sin(vecs.unsqueeze(0), v_pos)
        # print('cos_a2: ', cos_a2.shape, 'sin_a2: ', sin_a2.shape)

        ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
        rpe = torch.cat([ang_rpe, pos_rpe], dim=0)
        return rpe, mask

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        batch = from_numpy(batch)
        data = dict()
        data['BATCH_SIZE'] = len(batch)
        # Batching by use a list for non-fixed size
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
        '''
            Keys:
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS', 'LANE_GRAPH', 'RPE'
        '''

        actors, actor_idcs = self.actor_gather(data['BATCH_SIZE'], data['TRAJS'])
        lanes, lane_idcs = self.graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])

        data['ACTORS'] = actors
        data['ACTOR_IDCS'] = actor_idcs
        data['LANES'] = lanes
        data['LANE_IDCS'] = lane_idcs
        return data

    def actor_gather(self, batch_size, trajs):
        num_actors = [len(x['TRAJS_CTRS']) for x in trajs]
        # print('num_actors: ', num_actors)

        act_feats = []
        for i in range(batch_size):
            traj_pos = trajs[i]['TRAJS_POS_OBS']
            traj_disp = torch.zeros_like(traj_pos)
            traj_disp[:, 1:, :] = traj_pos[:, 1:, :] - traj_pos[:, :-1, :]

            act_feat = torch.cat([traj_disp,
                                  trajs[i]['TRAJS_ANG_OBS'],
                                  trajs[i]['TRAJS_VEL_OBS'],
                                  trajs[i]['TRAJS_TYPE'],
                                  trajs[i]['PAD_OBS'].unsqueeze(-1)], dim=-1)
            # print('act_feat: ', act_feat.shape)  # [N_a, 50, 14]
            act_feats.append(act_feat)

        act_feats = [x.transpose(1, 2) for x in act_feats]
        actors = torch.cat(act_feats, 0)  # [N_a, feat_len, 50], N_a is agent number in a batch
        actors = actors[..., 2:]  # ! tmp solution
        actor_idcs = []  # e.g. [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])]
        count = 0
        for i in range(batch_size):
            idcs = torch.arange(count, count + num_actors[i])
            actor_idcs.append(idcs)
            count += num_actors[i]
        # print('actor_idcs: ', actor_idcs)
        return actors, actor_idcs

    def graph_gather(self, batch_size, graphs):
        '''
            graphs[i]
                node_ctrs           torch.Size([116, N_{pt}, 2])
                node_vecs           torch.Size([116, N_{pt}, 2])
                intersect           torch.Size([116, N_{pt}])
                lane_type           torch.Size([116, N_{pt}, 3])
                cross_left          torch.Size([116, N_{pt}, 3])
                cross_right         torch.Size([116, N_{pt}, 3])
                left                torch.Size([116, N_{pt}])
                right               torch.Size([116, N_{pt}])
                lane_ctrs           torch.Size([116, 2])
                lane_vecs           torch.Size([116, 2])
                num_nodes           1160
                num_lanes           116
        '''
        lane_idcs = list()
        lane_count = 0
        for i in range(batch_size):
            l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            lane_idcs.append(l_idcs)
            lane_count = lane_count + graphs[i]["num_lanes"]
        # print('lane_idcs: ', lane_idcs)

        graph = dict()
        for key in ["node_ctrs", "node_vecs", "intersect", "lane_type", "cross_left", "cross_right", "left", "right"]:
            graph[key] = torch.cat([x[key] for x in graphs], 0)
            # print(key, graph[key].shape)
        for key in ["lane_ctrs", "lane_vecs"]:
            graph[key] = [x[key] for x in graphs]
            # print(key, [x.shape for x in graph[key]])

        lanes = torch.cat([graph['node_ctrs'],
                           graph['node_vecs'],
                           graph['intersect'].unsqueeze(2),
                           graph['lane_type'],
                           graph['cross_left'],
                           graph['cross_right'],
                           graph['left'].unsqueeze(2),
                           graph['right'].unsqueeze(2)], dim=-1)  # [N_{lane}, 9, F]
        # print('lanes: ', lanes.shape)
        return lanes, lane_idcs

    def rpe_gather(self, rpes):
        rpe = dict()
        for key in list(rpes[0].keys()):
            rpe[key] = [x[key] for x in rpes]
        # for k, v in rpe.items():
        #     print(k, len(v), [x.shape for x in v])
        return rpe

    def data_augmentation(self, df):
        '''
            "SEQ_ID", "CITY_NAME", "ORIG", "ROT",
            "TIMESTAMP", "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS", "LANE_GRAPH"

            "node_ctrs", "node_vecs",
            "turn", "control", "intersect", "left", "right"
            "lane_ctrs", "lane_vecs"
            "num_nodes", "num_lanes", "node_idcs", "lane_idcs"
        '''

        data = {}
        for key in list(df.keys()):
            data[key] = df[key].values[0]

        is_aug = random.choices([True, False], weights=[0.3, 0.7])[0]
        if not (self.aug and is_aug):
            return data

        # ~ random vertical flip
        data['TRAJS']['trajs_ctrs'][..., 1] *= -1
        data['TRAJS']['trajs_vecs'][..., 1] *= -1
        data['TRAJS']['trajs_pos'][..., 1] *= -1
        data['TRAJS']['trajs_ang'] *= -1
        data['TRAJS']['trajs_vel'][..., 1] *= -1

        data['LANE_GRAPH']['lane_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['lane_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['node_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['node_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['left'], data['LANE_GRAPH']['right'] = data['LANE_GRAPH']['right'], data['LANE_GRAPH']['left']

        return data
