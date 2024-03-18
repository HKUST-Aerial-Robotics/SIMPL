import os
#
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
#
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory
#
import init_path

_ESTIMATED_VEHICLE_LENGTH_M = 5.0
_ESTIMATED_VEHICLE_WIDTH_M = 2.0
_ESTIMATED_CYCLIST_LENGTH_M = 2.0
_ESTIMATED_CYCLIST_WIDTH_M = 0.7
_ESTIMATED_PEDESTRIAN_LENGTH_M = 0.3
_ESTIMATED_PEDESTRIAN_WIDTH_M = 0.5
_ESTIMATED_BUS_LENGTH_M = 7.0
_ESTIMATED_BUS_WIDTH_M = 2.1


class ArgoPreprocAV2():
    def __init__(self, args, verbose=False):
        self.args = args
        self.verbose = verbose
        self.debug = args.debug
        self.viz = args.viz
        self.mode = args.mode

        self.FAR_DIST_THRES = 20.0

        self.SEG_LENGTH = 15.0  # approximated lane segment length
        self.SEG_N_NODE = 10

        if self.debug:
            # self.map_vis = ArgoMapVisualizer()
            pass

    def print(self, info):
        if self.verbose:
            print(info)

    def process(self,
                seq_id: str,
                scenario: ArgoverseScenario,
                static_map: ArgoverseStaticMap):
        city_name = scenario.city_name

        # ~ get trajectories
        """
            trajs_pos   [N, 110(50), 2]
            trajs_ang   [N, 110(50)]
            trajs_vel   [N, 110(50), 2]
            trajs_type  [N, 110(50), 7]
            has_flags   [N, 110(50)]
        """
        trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags, trajs_tid, trajs_cat = self.get_trajectories(
            scenario, static_map)

        # ~ get origin and rot
        orig_seq, rot_seq, theta_seq = self.get_origin_rotation(trajs_pos[0], trajs_ang[0])  # * target-centric
        # ~ normalize w.r.t. scene
        trajs_pos = (trajs_pos - orig_seq).dot(rot_seq)
        trajs_ang = trajs_ang - theta_seq
        trajs_vel = trajs_vel.dot(rot_seq)

        # ~ normalize trajs
        trajs_pos_norm = []
        trajs_ang_norm = []
        trajs_vel_norm = []
        trajs_ctrs = []
        trajs_vecs = []
        for traj_pos, traj_ang, traj_vel in zip(trajs_pos, trajs_ang, trajs_vel):
            orig_act, rot_act, theta_act = self.get_origin_rotation(traj_pos, traj_ang)
            #
            trajs_pos_norm.append((traj_pos - orig_act).dot(rot_act))
            trajs_ang_norm.append(traj_ang - theta_act)
            trajs_vel_norm.append(traj_vel.dot(rot_act))
            #
            trajs_ctrs.append(orig_act)
            trajs_vecs.append(np.array([np.cos(theta_act), np.sin(theta_act)]))

        trajs_pos = np.array(trajs_pos_norm)  # [N, 110(50), 2]
        trajs_ang = np.array(trajs_ang_norm)  # [N, 110(50)]
        trajs_vel = np.array(trajs_vel_norm)  # [N, 110(50), 2]
        trajs_ctrs = np.array(trajs_ctrs)     # [N, 2]
        trajs_vecs = np.array(trajs_vecs)     # [N, 2]

        trajs = dict()
        trajs["trajs_pos"] = trajs_pos
        trajs["trajs_ang"] = trajs_ang
        trajs["trajs_vel"] = trajs_vel
        trajs["trajs_ctrs"] = trajs_ctrs
        trajs["trajs_vecs"] = trajs_vecs
        trajs["trajs_type"] = trajs_type
        trajs["has_flags"] = has_flags
        trajs["trajs_tid"] = trajs_tid
        trajs["trajs_cat"] = trajs_cat

        # ~ get lane graph
        lane_graph = self.get_lane_graph(seq_id, orig_seq, rot_seq, static_map)

        # collect data
        data = [[seq_id, city_name, orig_seq, rot_seq, trajs, lane_graph]]
        headers = ["SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TRAJS", "LANE_GRAPH"]

        # ! For debug
        if self.debug:
            print("[Debug]", seq_id, city_name)
            if self.viz:
                _, ax = plt.subplots(figsize=(10, 10))
                ax.axis('equal')
                vis_map = False
                self.plot_trajs(ax, trajs, orig_seq, rot_seq, vis_map=vis_map)
                self.plot_lane_graph(ax, seq_id, orig_seq, rot_seq, lane_graph, vis_map=vis_map)
                ax.set_title("{} {}".format(seq_id, city_name))
                plt.show()

        return data, headers

    def get_origin_rotation(self, traj_pos, traj_ang):
        orig = traj_pos[self.args.obs_len - 1]
        theta = traj_ang[self.args.obs_len - 1]
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        return orig, rot, theta

    def get_trajectories(self, scenario, static_map):
        # * find idcs
        focal_idx, av_idx = None, None
        scored_idcs, unscored_idcs, fragment_idcs = list(), list(), list()  # exclude AV
        for idx, x in enumerate(scenario.tracks):
            if x.track_id == scenario.focal_track_id and x.category == TrackCategory.FOCAL_TRACK:
                focal_idx = idx
            elif x.track_id == 'AV':
                av_idx = idx
            elif x.category == TrackCategory.SCORED_TRACK:
                scored_idcs.append(idx)
            elif x.category == TrackCategory.UNSCORED_TRACK:
                unscored_idcs.append(idx)
            elif x.category == TrackCategory.TRACK_FRAGMENT:
                fragment_idcs.append(idx)

        assert av_idx is not None, '[ERROR] Wrong av_idx'
        assert focal_idx is not None, '[ERROR] Wrong focal_idx'
        assert av_idx not in unscored_idcs, '[ERROR] Duplicated av_idx'

        sorted_idcs = [focal_idx, av_idx] + scored_idcs + unscored_idcs + fragment_idcs
        sorted_cat = ["focal", "av"] + ["score"] * \
            len(scored_idcs) + ["unscore"] * len(unscored_idcs) + ["frag"] * len(fragment_idcs)
        sorted_tid = [scenario.tracks[idx].track_id for idx in sorted_idcs]
        # print(focal_idx, av_idx, scored_idcs, unscored_idcs)
        # print('total: ', len(scenario.tracks),
        #       'focal & av: [{}, {}]'.format(focal_idx,  av_idx),
        #       'scored_idcs: ', len(scored_idcs),
        #       'unscored_idcs: ', len(unscored_idcs),
        #       'fragment_idcs: ', len(fragment_idcs))
        # print(len(sorted_idcs), sorted_idcs)
        # print(len(sorted_cat), sorted_cat)
        # print(len(sorted_tid), sorted_tid)
        # ! NOTICE: focal, av and scored are fully observed

        # * get timestamps and timesteps
        timestamps = scenario.timestamps_ns  # 100 ms interval
        if not self.mode == 'test':
            ts = np.arange(0, self.args.obs_len + self.args.pred_len)  # [0, 1,..., 109]
        else:
            ts = np.arange(0, self.args.obs_len)  # [0, 1,..., 49]
        ts_obs = ts[self.args.obs_len - 1]  # always 49

        # ~ Get map points
        map_pts = []
        for lane_id, lane in static_map.vector_lane_segments.items():
            # get lane centerline
            cl = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]  # use xy
            # print(cl.shape)
            map_pts.append(cl)
        map_pts = np.concatenate(map_pts, axis=0)  # [N_{map}, 2]
        map_pts = np.expand_dims(map_pts, axis=0)  # [1, N_{map}, 2]

        # * must follows the pre-defined order
        trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags = list(), list(), list(), list(), list()
        trajs_tid, trajs_cat = list(), list()  # track id and category
        for k, ind in enumerate(sorted_idcs):
            track = scenario.tracks[ind]

            traj_ts = np.array([x.timestep for x in track.object_states], dtype=np.int16)  # [N_{frames}]
            traj_pos = np.array([list(x.position) for x in track.object_states])  # [N_{frames}, 2]
            traj_ang = np.array([x.heading for x in track.object_states])  # [N_{frames}]
            traj_vel = np.array([list(x.velocity) for x in track.object_states])  # [N_{frames}, 2]

            # * only contains future part
            if traj_ts[0] > ts_obs:
                continue
            # * not observed at ts_obs
            if ts_obs not in traj_ts:
                continue

            # * far away from map (only for observed part)
            traj_obs_pts = np.expand_dims(traj_pos[:self.args.obs_len], axis=1)  # [N_{frames}, 1, 2]
            dist = np.linalg.norm(traj_obs_pts - map_pts, axis=-1)  # [N_{frames}, N_{map}]
            if np.min(dist) > self.FAR_DIST_THRES and sorted_cat[k] in ['unscore', 'frag']:
                continue

            has_flag = np.zeros_like(ts)
            # print(has_flag.shape, traj_ts.shape, traj_ts)
            has_flag[traj_ts] = 1

            # object type
            obj_type = np.zeros(7)  # 7 types
            if track.object_type == ObjectType.VEHICLE:
                obj_type[0] = 1
            elif track.object_type == ObjectType.PEDESTRIAN:
                obj_type[1] = 1
            elif track.object_type == ObjectType.MOTORCYCLIST:
                obj_type[2] = 1
            elif track.object_type == ObjectType.CYCLIST:
                obj_type[3] = 1
            elif track.object_type == ObjectType.BUS:
                obj_type[4] = 1
            elif track.object_type == ObjectType.UNKNOWN:
                obj_type[5] = 1
            else:
                obj_type[6] = 1  # for all static objects
            traj_type = np.zeros((len(ts), 7))
            traj_type[traj_ts] = obj_type

            # pad pos, nearest neighbor
            traj_pos_pad = np.full((len(ts), 2), None)
            traj_pos_pad[traj_ts] = traj_pos
            traj_pos_pad = self.padding_traj_nn(traj_pos_pad)
            # pad ang, nearest neighbor
            traj_ang_pad = np.full(len(ts), None)
            traj_ang_pad[traj_ts] = traj_ang
            traj_ang_pad = self.padding_traj_nn(traj_ang_pad)
            # pad vel, fill zeros
            traj_vel_pad = np.full((len(ts), 2), 0.0)
            traj_vel_pad[traj_ts] = traj_vel

            trajs_pos.append(traj_pos_pad)
            trajs_ang.append(traj_ang_pad)
            trajs_vel.append(traj_vel_pad)
            trajs_type.append(traj_type)
            has_flags.append(has_flag)
            trajs_tid.append(sorted_tid[k])
            trajs_cat.append(sorted_cat[k])

        # print('after filtering: {}/{}'.format(len(has_flags), len(scenario.tracks)))
        # _, ax = plt.subplots()
        # ax.axis('equal')
        # for idx in sorted_idcs:
        #     track = scenario.tracks[idx]
        #     traj_pos = np.array([list(x.position) for x in track.object_states])  # [N_{frames}, 2]
        #     ax.plot(traj_pos[:, 0], traj_pos[:, 1], color='grey', alpha=0.5)
        # for traj, angs, vels in zip(trajs_pos, trajs_ang, trajs_vel):
        #     ax.plot(traj[:, 0], traj[:, 1], color='orange', alpha=0.5, marker='.', zorder=10)
        #     for pt, ang, vel in zip(traj, angs, vels):
        #         ax.arrow(pt[0], pt[1], vel[0], vel[1], alpha=0.2, color='green')
        #         vec = np.array([np.cos(ang), np.sin(ang)])
        #         ax.arrow(pt[0], pt[1], vec[0], vec[1], alpha=0.2, color='cyan', zorder=5)
        # # plt.show()

        trajs_pos = np.array(trajs_pos).astype(np.float32)  # [N, 110(50), 2]
        trajs_ang = np.array(trajs_ang).astype(np.float32)  # [N, 110(50)]
        trajs_vel = np.array(trajs_vel).astype(np.float32)  # [N, 110(50), 2]
        trajs_type = np.array(trajs_type).astype(np.int16)  # [N, 110(50), 7]
        has_flags = np.array(has_flags).astype(np.int16)    # [N, 110(50)]
        # print('trajs_tid: ', len(trajs_tid), trajs_tid)   # List[str]
        # print('trajs_cat: ', len(trajs_cat), trajs_cat)   # List[str]

        # # double check focal and scored tracks are preserved
        # scored_tid = []
        # for idx, x in enumerate(scenario.tracks):
        #     if x.track_id == scenario.focal_track_id:
        #         assert x.track_id == trajs_tid[0], '[ERROR] Wrong focal track id'
        #     elif x.category == TrackCategory.SCORED_TRACK:
        #         scored_tid.append(x.track_id)
        # print('scored_tid: ', scored_tid)
        # idcs = [i for i, x in enumerate(trajs_cat) if x == 'score']
        # found_tid = [trajs_tid[i] for i in idcs]
        # print('found_tid: ', found_tid)
        # assert sorted(found_tid) == sorted(scored_tid), '[ERROR] scored track not consistent'

        return trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags, trajs_tid, trajs_cat

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

    def get_lane_graph(self,
                       seq_id: str,
                       orig: np.ndarray,
                       rot: np.ndarray,
                       static_map: ArgoverseStaticMap):
        node_ctrs, node_vecs, lane_type, intersect, cross_left, cross_right, left, right = [], [], [], [], [], [], [], []
        lane_ctrs, lane_vecs = [], []
        NUM_SEG_POINTS = 10

        for lane_id, lane in static_map.vector_lane_segments.items():
            # get lane centerline
            cl_raw = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]  # use xy
            assert cl_raw.shape[0] == NUM_SEG_POINTS, "[Error] Wrong num of points in lane - {}:{}".format(
                lane_id, cl_raw.shape[0])

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
                ctrln = np.array(LineString(cl_pts).coords)  # [num_sub_segs + 1, 2]
                ctrln = (ctrln - orig).dot(rot)  # to local frame

                anch_pos = np.mean(ctrln, axis=0)
                anch_vec = (ctrln[-1] - ctrln[0]) / np.linalg.norm(ctrln[-1] - ctrln[0])
                anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                     [anch_vec[1], anch_vec[0]]])

                lane_ctrs.append(anch_pos)
                lane_vecs.append(anch_vec)

                ctrln = (ctrln - anch_pos).dot(anch_rot)  # to instance frame

                ctrs = np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)
                vecs = np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
                node_ctrs.append(ctrs)  # middle point
                node_vecs.append(vecs)

                # ~ lane type
                lane_type_tmp = np.zeros(3)
                if lane.lane_type == LaneType.VEHICLE:
                    lane_type_tmp[0] = 1
                elif lane.lane_type == LaneType.BIKE:
                    lane_type_tmp[1] = 1
                elif lane.lane_type == LaneType.BUS:
                    lane_type_tmp[2] = 1
                else:
                    assert False, "[Error] Wrong lane type"
                lane_type.append(np.expand_dims(lane_type_tmp, axis=0).repeat(num_sub_segs, axis=0))

                # ~ intersection
                if lane.is_intersection:
                    intersect.append(np.ones(num_sub_segs, np.float32))
                else:
                    intersect.append(np.zeros(num_sub_segs, np.float32))

                # ~ lane marker type
                cross_left_tmp = np.zeros(3)
                if lane.left_mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                           LaneMarkType.DASH_SOLID_WHITE,
                                           LaneMarkType.DASHED_WHITE,
                                           LaneMarkType.DASHED_YELLOW,
                                           LaneMarkType.DOUBLE_DASH_YELLOW,
                                           LaneMarkType.DOUBLE_DASH_WHITE]:
                    cross_left_tmp[0] = 1  # crossable
                elif lane.left_mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW,
                                             LaneMarkType.DOUBLE_SOLID_WHITE,
                                             LaneMarkType.SOLID_YELLOW,
                                             LaneMarkType.SOLID_WHITE,
                                             LaneMarkType.SOLID_DASH_WHITE,
                                             LaneMarkType.SOLID_DASH_YELLOW,
                                             LaneMarkType.SOLID_BLUE]:
                    cross_left_tmp[1] = 1  # not crossable
                else:
                    cross_left_tmp[2] = 1  # unknown/none

                cross_right_tmp = np.zeros(3)
                if lane.right_mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                            LaneMarkType.DASH_SOLID_WHITE,
                                            LaneMarkType.DASHED_WHITE,
                                            LaneMarkType.DASHED_YELLOW,
                                            LaneMarkType.DOUBLE_DASH_YELLOW,
                                            LaneMarkType.DOUBLE_DASH_WHITE]:
                    cross_right_tmp[0] = 1  # crossable
                elif lane.right_mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW,
                                              LaneMarkType.DOUBLE_SOLID_WHITE,
                                              LaneMarkType.SOLID_YELLOW,
                                              LaneMarkType.SOLID_WHITE,
                                              LaneMarkType.SOLID_DASH_WHITE,
                                              LaneMarkType.SOLID_DASH_YELLOW,
                                              LaneMarkType.SOLID_BLUE]:
                    cross_right_tmp[1] = 1  # not crossable
                else:
                    cross_right_tmp[2] = 1  # unknown/none

                cross_left.append(np.expand_dims(cross_left_tmp, axis=0).repeat(num_sub_segs, axis=0))
                cross_right.append(np.expand_dims(cross_right_tmp, axis=0).repeat(num_sub_segs, axis=0))

                # ~ has left/right neighbor
                if lane.left_neighbor_id is None:
                    left.append(np.zeros(num_sub_segs, np.float32))  # w/o left neighbor
                else:
                    left.append(np.ones(num_sub_segs, np.float32))
                if lane.right_neighbor_id is None:
                    right.append(np.zeros(num_sub_segs, np.float32))  # w/o right neighbor
                else:
                    right.append(np.ones(num_sub_segs, np.float32))

        node_idcs = []  # List of range
        count = 0
        for i, ctr in enumerate(node_ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)

        lane_idcs = []  # node belongs to which lane, e.g. [0   0   0 ... 122 122 122]
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int16))
        # print("lane_idcs: ", lane_idcs.shape, lane_idcs)

        graph = dict()
        # geometry
        graph['node_ctrs'] = np.stack(node_ctrs, axis=0).astype(np.float32)
        graph['node_vecs'] = np.stack(node_vecs, axis=0).astype(np.float32)
        graph['lane_ctrs'] = np.array(lane_ctrs).astype(np.float32)
        graph['lane_vecs'] = np.array(lane_vecs).astype(np.float32)
        # node features
        graph['lane_type'] = np.stack(lane_type, axis=0).astype(np.int16)
        graph['intersect'] = np.stack(intersect, axis=0).astype(np.int16)
        graph['cross_left'] = np.stack(cross_left, axis=0).astype(np.int16)
        graph['cross_right'] = np.stack(cross_right, axis=0).astype(np.int16)
        graph['left'] = np.stack(left, axis=0).astype(np.int16)
        graph['right'] = np.stack(right, axis=0).astype(np.int16)

        # for k, v in graph.items():
        #     # node_ctrs, node_vecs, lane_type, intersect, cross_left, cross_right, left, right
        #     print(k, v.shape)

        # # node - lane
        graph['num_nodes'] = graph['node_ctrs'].shape[0] * graph['node_ctrs'].shape[1]
        graph['num_lanes'] = graph['lane_ctrs'].shape[0]
        # print('nodes: {}, lanes: {}'.format(graph['num_nodes'], graph['num_lanes']))
        return graph

    # plotters
    def plot_trajs(self, ax, trajs, orig, rot, vis_map=True):
        if not vis_map:
            rot = np.eye(2)
            orig = np.zeros(2)

        trajs_ctrs = trajs["trajs_ctrs"]
        trajs_vecs = trajs["trajs_vecs"]
        trajs_pos = trajs["trajs_pos"]
        trajs_ang = trajs["trajs_ang"]
        trajs_vel = trajs["trajs_vel"]
        trajs_type = trajs["trajs_type"]
        has_flags = trajs["has_flags"]
        trajs_tid = trajs['trajs_tid']
        trajs_cat = trajs['trajs_cat']

        # print('trajs_ctrs: ', trajs_ctrs.shape)
        # print('trajs_vecs: ', trajs_vecs.shape)
        # print('trajs_pos: ', trajs_pos.shape)
        # print('trajs_ang: ', trajs_ang.shape)
        # print('trajs_vel: ', trajs_vel.shape)
        # print('trajs_type: ', trajs_type.shape)
        # print('has_flags: ', has_flags.shape)
        # print('trajs_tid: ', len(trajs_tid))
        # print('trajs_cat: ', len(trajs_cat))

        assert np.all(has_flags[:, self.args.obs_len - 1] == 1), "[Error] Wrong has_flags"

        for i, (traj, ctr, vec) in enumerate(zip(trajs_pos, trajs_ctrs, trajs_vecs)):
            zorder = 10
            obj_cat = trajs_cat[i]
            if obj_cat == 'focal':
                clr = 'r'
                zorder = 20
            elif obj_cat == 'av':
                clr = 'green'
                zorder = 15
            elif obj_cat == 'score':
                clr = 'orange'
            elif obj_cat == 'unscore':
                clr = 'navy'
            elif obj_cat == 'frag':
                clr = 'grey'
            else:
                assert False, "[Error] Wrong category: {}".format(obj_cat)

            theta = np.arctan2(vec[1], vec[0])
            act_rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            traj = traj.dot(act_rot.T) + ctr
            traj = traj.dot(rot.T) + orig

            # vis traj
            ax.plot(traj[:self.args.obs_len, 0], traj[:self.args.obs_len, 1],
                    linewidth=3, alpha=0.5, color=clr, zorder=zorder)
            if not self.mode == 'test':
                ax.plot(traj[self.args.obs_len:, 0], traj[self.args.obs_len:, 1],
                        marker='.', alpha=0.5, color=clr, zorder=zorder)
            ax.plot(traj[self.args.obs_len - 1, 0], traj[self.args.obs_len - 1, 1],
                    marker='s', alpha=0.5, color=clr, zorder=zorder)
            ax.scatter(traj[:, 0], traj[:, 1], s=list((1 - has_flags[i])
                       * 50 + 1), color='darkgreen', alpha=0.2, zorder=5)

            ax.text(traj[self.args.obs_len, 0], traj[self.args.obs_len, 1], '{}:{}'.format(i, trajs_tid[i]))
            ax.arrow(ctr[0], ctr[1], vec[0], vec[1], alpha=0.5, color=clr, width=0.05, zorder=zorder)

            # attrs
            traj_type = np.where(trajs_type[i][self.args.obs_len - 1])[0][0]
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
                bbox_l = 1.0
                bbox_w = 1.0
            else:
                bbox_l = 1.0
                bbox_w = 1.0
            bbox = np.array([[-bbox_l / 2, -bbox_w / 2],
                             [-bbox_l / 2, bbox_w / 2],
                             [bbox_l / 2, bbox_w / 2],
                             [bbox_l / 2, -bbox_w / 2]])
            bbox = bbox.dot(act_rot.T) + ctr
            bbox = bbox.dot(rot.T) + orig
            ax.fill(bbox[:, 0], bbox[:, 1], color=clr, alpha=0.5, zorder=zorder)

    def plot_lane_graph(self, ax, seq_id, orig, rot, lane_graph, vis_map=True):
        if vis_map:
            pass
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
                         alpha=0.1,
                         width=0.1)

            anch_pos = anch_pos.dot(rot.T) + orig
            anch_vec = anch_vec.dot(rot.T)
            ax.plot(anch_pos[0], anch_pos[1], marker='o', color='blue', alpha=0.1)
            ax.arrow(anch_pos[0], anch_pos[1], anch_vec[0], anch_vec[1], alpha=0.1, color='blue', width=0.05)

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
                             alpha=0.1,
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
                             alpha=0.1,
                             width=0.05)
