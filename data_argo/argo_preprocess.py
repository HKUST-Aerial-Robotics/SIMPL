import os
from os.path import expanduser
import time
from typing import Any, Dict, List, Tuple
import random
import copy
#
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union, nearest_points, unary_union
from scipy import sparse, spatial
#
from argoverse.map_representation.map_api import ArgoverseMap
#
import init_path
from utils.vis_utils import ArgoMapVisualizer


class ArgoPreproc():
    def __init__(self, args, verbose=False):
        self.args = args
        self.verbose = verbose
        self.debug = args.debug
        self.viz = args.viz
        self.mode = args.mode

        self.MAP_RADIUS = 80.0
        self.COMPL_RANGE = 30.0
        self.REL_LANE_THRES = 30.0

        self.SEG_LENGTH = 10.0  # approximated lane segment length
        self.SEG_N_NODE = 10

        self.argo_map = ArgoverseMap()

        if self.debug:
            self.map_vis = ArgoMapVisualizer()

    def print(self, info):
        if self.verbose:
            print(info)

    def process(self, seq_id, df):
        city_name = df['CITY_NAME'].values[0]

        # get trajectories
        ts, trajs_ori, pad_flags = self.get_trajectories(df, city_name)

        # get origin and rot
        orig, rot = self.get_origin_rotation(trajs_ori[0])  # * ego-centric
        trajs_ori = (trajs_ori - orig).dot(rot)

        # ~ normalize trajs
        trajs_norm = []
        trajs_ctrs = []
        trajs_vecs = []
        for traj in trajs_ori:
            act_orig = traj[self.args.obs_len - 1]
            act_vec = act_orig - traj[0]
            theta = np.arctan2(act_vec[1], act_vec[0])
            act_rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            trajs_norm.append((traj - act_orig).dot(act_rot))
            trajs_ctrs.append(act_orig)
            trajs_vecs.append(np.array([np.cos(theta), np.sin(theta)]))

        trajs = np.array(trajs_norm)
        trajs_ctrs = np.array(trajs_ctrs)
        trajs_vecs = np.array(trajs_vecs)

        # get ROI
        lane_ids = self.get_related_lanes(seq_id, city_name, orig, expand_dist=self.MAP_RADIUS)
        # get lane graph
        lane_graph = self.get_lane_graph(seq_id, df, city_name, orig, rot, lane_ids)

        # find relevant lanes
        trajs_ori_flat = trajs_ori.reshape(-1, 2)
        lane_ctrs = lane_graph['lane_ctrs']
        # print("trajs_ori_flat: ", trajs_ori_flat.shape, "lane_ctrs: ", lane_ctrs.shape)
        dists = spatial.distance.cdist(trajs_ori_flat, lane_ctrs)  # calculate distance
        rel_lane_flags = np.min(dists, axis=0) < self.REL_LANE_THRES  # find lanes that are not close to the trajectory
        lane_graph['rel_lane_flags'] = rel_lane_flags  # [True False True ...]

        # collect data
        data = [[seq_id, city_name, orig, rot, ts, trajs, trajs_ctrs, trajs_vecs, pad_flags, lane_graph]]
        headers = ["SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TIMESTAMP",
                   "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS", "LANE_GRAPH"]

        # ! For debug
        if self.debug and self.viz:
            _, ax = plt.subplots(figsize=(10, 10))
            ax.axis('equal')
            vis_map = False
            self.plot_trajs(ax, trajs, trajs_ctrs, trajs_vecs, pad_flags, orig, rot, vis_map=vis_map)
            self.plot_lane_graph(ax, city_name, orig, rot, lane_ids, lane_graph, vis_map=vis_map)
            ax.set_title("{} {}".format(seq_id, city_name))
            plt.show()

        return data, headers

    def get_origin_rotation(self, traj):
        orig = traj[self.args.obs_len - 1]

        # vec = orig - traj[self.args.obs_len - 2]
        vec = orig - traj[0]
        theta = np.arctan2(vec[1], vec[0])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

        return orig, rot

    def get_trajectories(self,
                         df: pd.DataFrame,
                         city_name: str):
        ts = np.sort(np.unique(df['TIMESTAMP'].values)).astype(float)
        t_obs = ts[self.args.obs_len - 1]

        agent_traj = df[df["OBJECT_TYPE"] == "AGENT"]
        agent_traj = np.stack((agent_traj['X'].values, agent_traj['Y'].values), axis=1).astype(float)
        agent_traj[:, 0:2] = agent_traj[:, 0:2]

        av_traj = df[df["OBJECT_TYPE"] == "AV"]
        av_traj = np.stack((av_traj['X'].values, av_traj['Y'].values), axis=1).astype(float)
        av_traj[:, 0:2] = av_traj[:, 0:2]

        assert len(agent_traj) == len(av_traj), "Shape error for AGENT and AV, AGENT: {}, AV: {}".format(
            agent_traj.shape, av_traj.shape)

        trajs = [agent_traj, av_traj]
        pred_ctr = agent_traj[self.args.obs_len - 1]

        pad_flags = [np.ones_like(ts), np.ones_like(ts)]

        track_ids = np.unique(df["TRACK_ID"].values)
        for idx in track_ids:
            mot_traj = df[df["TRACK_ID"] == idx]

            if mot_traj['OBJECT_TYPE'].values[0] == 'AGENT' or mot_traj['OBJECT_TYPE'].values[0] == 'AV':
                continue

            ts_mot = np.array(mot_traj['TIMESTAMP'].values).astype(float)
            mot_traj = np.stack((mot_traj['X'].values, mot_traj['Y'].values), axis=1).astype(float)

            # ~ remove traj after t_obs
            if np.all(ts_mot > t_obs):
                continue

            _, idcs, _ = np.intersect1d(ts, ts_mot, return_indices=True)
            padded = np.zeros_like(ts)
            padded[idcs] = 1

            if not padded[self.args.obs_len - 1]:  # !
                continue

            mot_traj_pad = np.full(agent_traj[:, :2].shape, None)
            mot_traj_pad[idcs] = mot_traj

            mot_traj_pad = self.padding_traj_nn(mot_traj_pad)
            assert np.all(mot_traj_pad[idcs] == mot_traj), "Padding error"

            mot_traj = np.stack((mot_traj_pad[:, 0], mot_traj_pad[:, 1]), axis=1)
            mot_traj[:, 0:2] = mot_traj[:, 0:2]

            mot_ctr = mot_traj[self.args.obs_len - 1]
            if np.linalg.norm(mot_ctr - pred_ctr) > self.MAP_RADIUS:  # ! too far
                continue

            trajs.append(mot_traj)
            pad_flags.append(padded)

        ts = (ts - ts[0]).astype(np.float32)
        trajs = np.array(trajs).astype(np.float32)  # [N, 50(20), 2]
        pad_flags = np.array(pad_flags).astype(np.int16)  # [N, 50(20)]

        return ts, trajs, pad_flags

    def padding_traj_nn(self, traj):
        n = len(traj)

        # forward
        buff = None
        for i in range(n):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        # backward
        buff = None
        for i in reversed(range(n)):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        return traj

    def get_related_lanes(self, seq_id, city_name, orig, expand_dist):
        lane_ids = self.argo_map.get_lane_ids_in_xy_bbox(orig[0], orig[1], city_name, self.MAP_RADIUS)
        return copy.deepcopy(lane_ids)

    def get_lane_graph(self, seq_id, df, city_name, orig, rot, lane_ids):
        node_ctrs, node_vecs, turn, control, intersect, left, right = [], [], [], [], [], [], []
        lane_ctrs, lane_vecs = [], []

        lanes = self.argo_map.city_lane_centerlines_dict[city_name]
        for lane_id in lane_ids:
            lane = lanes[lane_id]

            cl_raw = lane.centerline
            cl_ls = LineString(cl_raw)

            num_segs = np.max([int(np.floor(cl_ls.length / self.SEG_LENGTH)), 1])
            ds = cl_ls.length / num_segs
            for i in range(num_segs):
                s_lb = i * ds
                s_ub = (i + 1) * ds
                num_sub_segs = self.SEG_N_NODE

                cl_pts = []
                for s in np.linspace(s_lb, s_ub, num_sub_segs + 1):
                    cl_pts.append(cl_ls.interpolate(s))
                ctrln = np.array(LineString(cl_pts).coords)
                ctrln[:, 0:2] = (ctrln[:, 0:2] - orig).dot(rot)

                anch_pos = np.mean(ctrln, axis=0)
                anch_vec = (ctrln[-1] - ctrln[0]) / np.linalg.norm(ctrln[-1] - ctrln[0])
                anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                     [anch_vec[1], anch_vec[0]]])

                lane_ctrs.append(anch_pos)
                lane_vecs.append(anch_vec)

                ctrln[:, 0:2] = (ctrln[:, 0:2] - anch_pos).dot(anch_rot)

                ctrs = np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)
                vecs = np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
                node_ctrs.append(ctrs)  # middle point
                node_vecs.append(vecs)

                # ~ has left/right neighbor
                if lane.l_neighbor_id is None:
                    # w/o left neighbor
                    left.append(np.zeros(num_sub_segs, np.float32))
                else:
                    left.append(np.ones(num_sub_segs, np.float32))

                if lane.r_neighbor_id is None:
                    # w/o right neighbor
                    right.append(np.zeros(num_sub_segs, np.float32))
                else:
                    right.append(np.ones(num_sub_segs, np.float32))

                # ~ turn dir
                x = np.zeros((num_sub_segs, 2), np.float32)
                if lane.turn_direction == 'LEFT':
                    x[:, 0] = 1
                elif lane.turn_direction == 'RIGHT':
                    x[:, 1] = 1
                else:
                    pass
                turn.append(x)

                # ~ control & intersection
                control.append(lane.has_traffic_control * np.ones(num_sub_segs, np.float32))
                intersect.append(lane.is_intersection * np.ones(num_sub_segs, np.float32))

        # for x in [node_ctrs, node_vecs, turn, control, intersect, left, right]:
        #     print([y.shape for y in node_ctrs])

        node_idcs = []  # List of range
        count = 0
        for i, ctr in enumerate(node_ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        num_lanes = len(node_idcs)

        lane_idcs = []  # node belongs to which lane, e.g. [0   0   0 ... 122 122 122]
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int16))
        # lane_idcs = np.concatenate(lane_idcs, 0)
        # print("lane_idcs: ", lane_idcs.shape, lane_idcs)

        # lane contains which node
        for i in range(len(node_idcs)):
            node_idcs[i] = np.array(node_idcs[i])
        # node_idcs = np.concatenate(node_idcs, 0)
        # print("node_idcs: ", node_idcs.shape, node_idcs)

        graph = dict()
        # geometry
        graph['node_ctrs'] = np.stack(node_ctrs, 0).astype(np.float32)
        graph['node_vecs'] = np.stack(node_vecs, 0).astype(np.float32)
        # node features
        graph['turn'] = np.stack(turn, 0).astype(np.int16)
        graph['control'] = np.stack(control, 0).astype(np.int16)
        graph['intersect'] = np.stack(intersect, 0).astype(np.int16)
        graph['left'] = np.stack(left, 0).astype(np.int16)
        graph['right'] = np.stack(right, 0).astype(np.int16)
        #
        graph['lane_ctrs'] = np.array(lane_ctrs).astype(np.float32)
        graph['lane_vecs'] = np.array(lane_vecs).astype(np.float32)
        # for k, v in graph.items():
        #     print(k, v.shape)

        # node - lane
        graph['num_nodes'] = num_nodes
        graph['num_lanes'] = num_lanes
        # print('nodes: {}, lanes: {}'.format(num_nodes, num_lanes))
        return graph

    # plotters
    def plot_trajs(self, ax, trajs, trajs_ctrs, trajs_vecs, pad_flags, orig, rot, vis_map=True):
        if not vis_map:
            rot = np.eye(2)
            orig = np.zeros(2)

        for i, (traj, ctr, vec) in enumerate(zip(trajs, trajs_ctrs, trajs_vecs)):
            zorder = 10
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'green'
            else:
                clr = 'orange'

            theta = np.arctan2(vec[1], vec[0])
            act_rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            traj = traj.dot(act_rot.T) + ctr

            traj = traj.dot(rot.T) + orig
            ax.plot(traj[:, 0], traj[:, 1], marker='.', alpha=0.5, color=clr, zorder=zorder)
            ax.text(traj[self.args.obs_len, 0], traj[self.args.obs_len, 1], '{}'.format(i))
            ax.scatter(traj[:, 0], traj[:, 1], s=list((1 - pad_flags[i]) * 50 + 1), color='b')

    def plot_lane_graph(self, ax, city_name, orig, rot, lane_ids, lane_graph, vis_map=True):
        if vis_map:
            self.map_vis.show_map_with_lanes(ax, city_name, orig, lane_ids)
        else:
            rot = np.eye(2)
            orig = np.zeros(2)

        node_ctrs = lane_graph['node_ctrs']
        node_vecs = lane_graph['node_vecs']
        node_left = lane_graph['left']
        node_right = lane_graph['right']

        lane_ctrs = lane_graph['lane_ctrs']
        lane_vecs = lane_graph['lane_vecs']

        # print('lane_ctrs: ', lane_ctrs.shape)
        # print('lane_vecs: ', lane_vecs.shape)

        rel_lane_flags = lane_graph['rel_lane_flags']
        ax.plot(lane_ctrs[rel_lane_flags][:, 0], lane_ctrs[rel_lane_flags][:, 1], 'x', color='red', markersize=10)

        for ctrs_tmp, vecs_tmp, left_tmp, right_tmp, anch_pos, anch_vec in zip(node_ctrs, node_vecs,
                                                                               node_left, node_right,
                                                                               lane_ctrs, lane_vecs):
            anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                 [anch_vec[1], anch_vec[0]]])
            ctrs_tmp = ctrs_tmp.dot(anch_rot.T) + anch_pos
            ctrs_tmp = ctrs_tmp.dot(rot.T) + orig

            # ax.plot(ctrs_tmp[:, 0], ctrs_tmp[:, 1], marker='.', alpha=0.5)

            vecs_tmp = vecs_tmp.dot(anch_rot.T)
            vecs_tmp = vecs_tmp.dot(rot.T)

            for j in range(vecs_tmp.shape[0]):
                vec = vecs_tmp[j]
                pt0 = ctrs_tmp[j] - vec / 2
                pt1 = ctrs_tmp[j] + vec / 2
                ax.arrow(pt0[0],
                         pt0[1],
                         (pt1-pt0)[0],
                         (pt1-pt0)[1],
                         edgecolor=None,
                         color='grey',
                         alpha=0.3,
                         width=0.1)

            anch_pos = anch_pos.dot(rot.T) + orig
            anch_vec = anch_vec.dot(rot.T)
            ax.plot(anch_pos[0], anch_pos[1], marker='*', color='cyan')
            ax.arrow(anch_pos[0], anch_pos[1], anch_vec[0], anch_vec[1], alpha=0.5, color='r', width=0.1)

            for i in range(len(left_tmp)):
                if left_tmp[i]:
                    ctr = ctrs_tmp[i]
                    vec = vecs_tmp[i] / np.linalg.norm(vecs_tmp[i])
                    vec = np.array([-vec[1], vec[0]])
                    ax.arrow(ctr[0],
                             ctr[1],
                             vec[0],
                             vec[1],
                             edgecolor=None,
                             color='red',
                             alpha=0.3,
                             width=0.05)

            for i in range(len(right_tmp)):
                if right_tmp[i]:
                    ctr = ctrs_tmp[i]
                    vec = vecs_tmp[i] / np.linalg.norm(vecs_tmp[i])
                    vec = np.array([vec[1], -vec[0]])
                    ax.arrow(ctr[0],
                             ctr[1],
                             vec[0],
                             vec[1],
                             edgecolor=None,
                             color='green',
                             alpha=0.3,
                             width=0.05)
