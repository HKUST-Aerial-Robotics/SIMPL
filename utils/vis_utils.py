from typing import Any, Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from pathlib import Path

from argoverse.map_representation.map_api import ArgoverseMap
#
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory


class ArgoMapVisualizer:
    def __init__(self):
        self.argo_map = ArgoverseMap()

    def show_lanes(self, ax, city_name, lane_ids, clr='g', alpha=0.2, show_lane_ids=False):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for idx in lane_ids:
            lane_cl = seq_lane_props[idx].centerline
            ax.plot(lane_cl[:, 0], lane_cl[:, 1], color=clr, alpha=alpha, linewidth=5)

            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

    def show_map_with_lanes(self,
                            ax,
                            city_name,
                            position,
                            lane_ids,
                            map_size=np.array([150.0, 150.0]),
                            show_freespace=True,
                            show_lane_ids=False):
        x_min = position[0] - map_size[0] / 2
        x_max = position[0] + map_size[0] / 2
        y_min = position[1] - map_size[1] / 2
        y_max = position[1] + map_size[1] / 2

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for idx in lane_ids:
            lane_cl = seq_lane_props[idx].centerline
            lane_polygon = self.argo_map.get_lane_segment_polygon(
                idx, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color='gray',
                        alpha=0.1,
                        edgecolor=None))

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]
            ax.arrow(pt[0],
                     pt[1],
                     vec[0],
                     vec[1],
                     alpha=0.5,
                     color='grey',
                     width=0.1,
                     zorder=1)
            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

        if show_freespace:
            drivable_area = self.argo_map.get_da_contours(city_name)
            surrounding_contours = []
            for contour in drivable_area:
                if (np.min(contour[:, 0]) < x_max
                        and np.min(contour[:, 1]) < y_max
                        and np.max(contour[:, 0]) > x_min
                        and np.max(contour[:, 1]) > y_min):
                    surrounding_contours.append(contour)

            for contour in surrounding_contours:
                ax.add_patch(
                    Polygon(contour[:, 0:2],
                            color='darkgray',
                            alpha=0.1,
                            edgecolor=None))

    def show_surrounding_elements(self,
                                  ax,
                                  city_name,
                                  position,
                                  map_size=np.array([150.0, 150.0]),
                                  show_freespace=True,
                                  show_lane_ids=False):
        x_min = position[0] - map_size[0] / 2
        x_max = position[0] + map_size[0] / 2
        y_min = position[1] - map_size[1] / 2
        y_max = position[1] + map_size[1] / 2

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]
        surrounding_lanes = {}
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max
                    and np.max(lane_cl[:, 0]) > x_min
                    and np.max(lane_cl[:, 1]) > y_min):
                surrounding_lanes[lane_id] = lane_cl

        for idx, lane_cl in surrounding_lanes.items():
            lane_polygon = self.argo_map.get_lane_segment_polygon(
                idx, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color='gray',
                        alpha=0.1,
                        edgecolor=None))

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]
            vec = vec / np.linalg.norm(vec) * 1.0
            # ax.arrow(pt[0],
            #          pt[1],
            #          vec[0],
            #          vec[1],
            #          alpha=0.5,
            #          color='grey',
            #          width=0.1,
            #          zorder=1)
            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

        if show_freespace:
            drivable_area = self.argo_map.get_da_contours(city_name)
            surrounding_contours = []
            for contour in drivable_area:
                if (np.min(contour[:, 0]) < x_max
                        and np.min(contour[:, 1]) < y_max
                        and np.max(contour[:, 0]) > x_min
                        and np.max(contour[:, 1]) > y_min):
                    surrounding_contours.append(contour)

            for contour in surrounding_contours:
                ax.add_patch(
                    Polygon(contour[:, 0:2],
                            color='darkgray',
                            alpha=0.1,
                            edgecolor=None))

    def show_all_map(self, ax, city_name, show_freespace=True):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]

            under_control = self.argo_map.lane_has_traffic_control_measure(
                lane_id, city_name)

            in_intersection = self.argo_map.lane_is_in_intersection(
                lane_id, city_name)

            turn_dir = self.argo_map.get_lane_turn_direction(
                lane_id, city_name)

            cl_clr = 'grey'

            if in_intersection:
                cl_clr = 'orange'

            if turn_dir == 'LEFT':
                cl_clr = 'blue'
            elif turn_dir == 'RIGHT':
                cl_clr = 'green'

            ax.arrow(pt[0],
                     pt[1],
                     vec[0],
                     vec[1],
                     alpha=0.5,
                     color=cl_clr,
                     width=0.1,
                     zorder=1)

            if under_control:
                p_vec = vec / np.linalg.norm(vec) * 1.5
                pt1 = pt + np.array([-p_vec[1], p_vec[0]])
                pt2 = pt + np.array([p_vec[1], -p_vec[0]])
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color='tomato',
                        alpha=0.5,
                        linewidth=2)

            lane_polygon = self.argo_map.get_lane_segment_polygon(
                lane_id, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color=cl_clr,
                        alpha=0.1,
                        edgecolor=None))

        if show_freespace:
            drivable_area = self.argo_map.get_da_contours(city_name)
            surrounding_contours = []
            for contour in drivable_area:
                surrounding_contours.append(contour)

            for contour in surrounding_contours:
                ax.add_patch(
                    Polygon(contour[:, 0:2],
                            color='darkgray',
                            alpha=0.1,
                            edgecolor=None))

    def transform_2d_gaussian(self, mu_x, mu_y, sig_x, sig_y, rho, rot_mat, pos):
        cov_xy = sig_x * sig_y * rho
        cov_mat = np.array([[sig_x**2, cov_xy], [cov_xy, sig_y**2]])

        _cov_mat = rot_mat.dot(cov_mat).dot(rot_mat.T)
        _sig_x = np.sqrt(_cov_mat[0, 0])
        _sig_y = np.sqrt(_cov_mat[1, 1])
        _cov_xy = _cov_mat[0, 1]
        _rho = _cov_xy / (_sig_x * _sig_y)

        _mu_tmp = rot_mat.dot(np.array([[mu_x], [mu_y]])).flatten()

        _mu_x = _mu_tmp[0] + pos[0]
        _mu_y = _mu_tmp[1] + pos[1]

        return _mu_x, _mu_y, _sig_x, _sig_y, _rho

    # Visualize 2d gaussian distribution
    def get_confidence_ellipse(self,
                               mu_x,
                               mu_y,
                               sig_x,
                               sig_y,
                               rho,
                               trans,
                               n_std=3,
                               facecolor='none',
                               edgecolor='red',
                               alpha=0.3):
        ell_radius_x = np.sqrt(1 + rho)
        ell_radius_y = np.sqrt(1 - rho)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          alpha=alpha)

        # multiply stdandard deviation with the
        # given number of standard deviations.
        scale_x = sig_x * n_std
        scale_y = sig_y * n_std

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mu_x, mu_y)

        ellipse.set_transform(transf + trans)

        return ellipse


class AV2MapVisualizer:
    def __init__(self):
        self.dataset_dir = os.path.expanduser('~') + '/data/dataset/argoverse2/'

    def show_map(self,
                 ax,
                 split: str,
                 seq_id: str,
                 show_freespace=True):

        # ax.set_facecolor("grey")

        static_map_path = Path(self.dataset_dir + f"/{split}/{seq_id}" + f"/log_map_archive_{seq_id}.json")
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        # ~ drivable area
        # print('num drivable areas: ', len(static_map.vector_drivable_areas),
        #       [x for x in static_map.vector_drivable_areas.keys()])
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.2)

        # ~ lane segments
        # print('num lane segs: ', len(static_map.vector_lane_segments),
        #       [x for x in static_map.vector_lane_segments.keys()])
        print('Num lanes: ', len(static_map.vector_lane_segments))
        for lane_segment in static_map.vector_lane_segments.values():
            # print('left pts: ', lane_segment.left_lane_boundary.xyz.shape,
            #       'right pts: ', lane_segment.right_lane_boundary.xyz.shape)

            if lane_segment.lane_type == 'VEHICLE':
                lane_clr = 'blue'
            elif lane_segment.lane_type == 'BIKE':
                lane_clr = 'green'
            elif lane_segment.lane_type == 'BUS':
                lane_clr = 'orange'
            else:
                assert False, "Wrong lane type"

            # if lane_segment.is_intersection:
            #     lane_clr = 'yellow'

            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color=lane_clr, alpha=0.1)

            for boundary in [lane_segment.left_lane_boundary, lane_segment.right_lane_boundary]:
                ax.plot(boundary.xyz[:, 0],
                        boundary.xyz[:, 1],
                        linewidth=1,
                        color='grey',
                        alpha=0.3)

            # cl = static_map.get_lane_segment_centerline(lane_segment.id)
            # ax.plot(cl[:, 0], cl[:, 1], linestyle='--', color='magenta', alpha=0.1)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate([pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)])
            # plt.plot(edge[:, 0], edge[:, 1], color='orange', alpha=0.75)
            ax.fill(edge[:, 0], edge[:, 1], color='orange', alpha=0.2)
            # for edge in [ped_xing.edge1, ped_xing.edge2]:
            #     ax.plot(edge.xyz[:, 0], edge.xyz[:, 1], color='orange', alpha=0.5, linestyle='dotted')

    def show_map_clean(self,
                       ax,
                       split: str,
                       seq_id: str,
                       show_freespace=True):

        # ax.set_facecolor("grey")

        static_map_path = Path(self.dataset_dir + f"/{split}/{seq_id}" + f"/log_map_archive_{seq_id}.json")
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        # ~ drivable area
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.2)

        # ~ lane segments
        print('Num lanes: ', len(static_map.vector_lane_segments))
        for lane_id, lane_segment in static_map.vector_lane_segments.items():
            lane_clr = 'grey'
            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color='whitesmoke', alpha=1.0, edgecolor=None, zorder=0)

            # centerline
            centerline = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]  # use xy
            ax.plot(centerline[:, 0], centerline[:, 1], alpha=0.1, color='grey', linestyle='dotted', zorder=1)

            # lane boundary
            for boundary, mark_type in [(lane_segment.left_lane_boundary.xyz, lane_segment.left_mark_type),
                                        (lane_segment.right_lane_boundary.xyz, lane_segment.right_mark_type)]:

                clr = None
                width = 1.0
                if mark_type in [LaneMarkType.DASH_SOLID_WHITE,
                                 LaneMarkType.DASHED_WHITE,
                                 LaneMarkType.DOUBLE_DASH_WHITE,
                                 LaneMarkType.DOUBLE_SOLID_WHITE,
                                 LaneMarkType.SOLID_WHITE,
                                 LaneMarkType.SOLID_DASH_WHITE]:
                    clr = 'white'
                    zorder = 3
                    width = width
                elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                   LaneMarkType.DASHED_YELLOW,
                                   LaneMarkType.DOUBLE_DASH_YELLOW,
                                   LaneMarkType.DOUBLE_SOLID_YELLOW,
                                   LaneMarkType.SOLID_YELLOW,
                                   LaneMarkType.SOLID_DASH_YELLOW]:
                    clr = 'gold'
                    zorder = 4
                    width = width * 1.1

                style = 'solid'
                if mark_type in [LaneMarkType.DASHED_WHITE,
                                 LaneMarkType.DASHED_YELLOW,
                                 LaneMarkType.DOUBLE_DASH_YELLOW,
                                 LaneMarkType.DOUBLE_DASH_WHITE]:
                    style = (0, (5, 10))  # loosely dashed
                elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                   LaneMarkType.DASH_SOLID_WHITE,
                                   LaneMarkType.DOUBLE_SOLID_YELLOW,
                                   LaneMarkType.DOUBLE_SOLID_WHITE,
                                   LaneMarkType.SOLID_YELLOW,
                                   LaneMarkType.SOLID_WHITE,
                                   LaneMarkType.SOLID_DASH_WHITE,
                                   LaneMarkType.SOLID_DASH_YELLOW]:
                    style = 'solid'

                if (clr is not None) and (style is not None):
                    ax.plot(boundary[:, 0],
                            boundary[:, 1],
                            color=clr,
                            alpha=1.0,
                            linewidth=width,
                            linestyle=style,
                            zorder=zorder)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate([pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)])
            ax.fill(edge[:, 0], edge[:, 1], color='yellow', alpha=0.1, edgecolor=None)
