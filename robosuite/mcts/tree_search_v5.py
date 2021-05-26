import os
import sys
import copy
import random

import numpy as np

# import rospy
# from moveit_msgs.srv import *
# from moveit_msgs.msg import *
# from shape_msgs.msg import *
# from geometry_msgs.msg import *
# from mujoco_moveit_connector.srv import *

from robosuite.mcts.util_v3 import *
# from robosuite.utils import transform_utils as tf
# import trimesh.proximity

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# import torch

from matplotlib import pyplot as plt


def check_stability(obj_idx, object_list, meshes, com1, volume1, margin=0.011):
    if "on" in object_list[obj_idx].logical_state:
        obj_name = object_list[obj_idx].logical_state["on"][0]
        support_obj_idx = get_obj_idx_by_name(object_list, obj_name)
        mesh2 = deepcopy(meshes[object_list[support_obj_idx].mesh_idx])
        mesh2.apply_transform(object_list[support_obj_idx].pose)

        closest_points, dists, surface_idx = trimesh.proximity.closest_point(mesh2, [com1])
        project_point2 = closest_points[0]
        safe_com = mesh2.face_normals[surface_idx[0]][2] > 0.999 and (project_point2[0] > mesh2.bounds[0][0] + margin) and (project_point2[0] < mesh2.bounds[1][0] - margin) and (project_point2[1] > mesh2.bounds[0][1] + margin) and (
                    project_point2[1] < mesh2.bounds[1][1] - margin)

        if safe_com:
            com1 = (com1 * volume1 + mesh2.center_mass * mesh2.volume) / (volume1 + mesh2.volume)
            volume1 += mesh2.volume
            return check_stability(support_obj_idx, object_list, meshes, com1, volume1)
        else:
            return False
    else:
        return True


def geometry_based_physical_checker(obj_list, action, next_obj_list, meshes, network=None):
    if action['type'] is 'pick':
        return True
    if action['type'] is 'place':
        pick_idx = get_held_object(obj_list)
        mesh1 = deepcopy(meshes[next_obj_list[pick_idx].mesh_idx])
        mesh1.apply_transform(next_obj_list[pick_idx].pose)
        com1 = deepcopy(mesh1.center_mass)
        volume1 = deepcopy(mesh1.volume)
        flag_ = check_stability(pick_idx, next_obj_list, meshes, com1, volume1)
        return flag_


def sampler(_exploration_method, _action_values, _visits, _depth, _indices=None, eps=0.):
    if _indices is not None:
        selected_action_values = [_action_values[_index] for _index in _indices]
        selected_visits = [_visits[_index] for _index in _indices]
    else:
        selected_action_values = deepcopy(_action_values)
        selected_visits = _visits

    selected_action_values = np.asarray(selected_action_values)
    selected_action_values[np.isinf(selected_action_values)] = 0.

    if eps > np.random.uniform() or _exploration_method['method'] is 'random':
        selected_idx = np.random.choice(len(selected_action_values), size=1)[0]
    elif _exploration_method['method'] is 'ucb':
        c = _exploration_method['param'] / np.maximum(_depth, 1)
        upper_confidence_bounds = selected_action_values + c * np.sqrt(1. / np.maximum(1., selected_visits))
        selected_idx = np.argmax(upper_confidence_bounds)
    elif _exploration_method['method'] is 'bai_ucb':
        if len(selected_visits) == 1:
            selected_idx = 0
        else:
            c = _exploration_method['param'] / np.maximum(_depth, 1)
            upper_bounds = selected_action_values + c * np.sqrt(1. / np.maximum(1., selected_visits))
            lower_bounds = selected_action_values - c * np.sqrt(1. / np.maximum(1., selected_visits))
            B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]
            b = np.argmin(B_k)
            u = np.argmax(upper_bounds)
            if selected_visits[b] > selected_visits[u]:
                selected_idx = u
            else:
                selected_idx = b
    elif _exploration_method['method'] is 'bai_perturb':
        if len(selected_visits) == 1:
            selected_idx = 0
        else:
            c = _exploration_method['param'] / np.maximum(_depth, 1)
            g = np.random.normal(size=(len(selected_visits)))
            upper_bounds = selected_action_values + c * np.sqrt(1. / np.maximum(1., selected_visits)) * g
            lower_bounds = selected_action_values - c * np.sqrt(1. / np.maximum(1., selected_visits)) * g
            B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]
            b = np.argmin(B_k)
            u = np.argmax(upper_bounds)
            if selected_visits[b] > selected_visits[u]:
                selected_idx = u
            else:
                selected_idx = b
    elif _exploration_method['method'] is 'greedy':
        selected_idx = np.argmax(selected_action_values)
    if _indices is not None:
        return _indices[selected_idx]
    else:
        return selected_idx


class Tree(object):
    def __init__(self,
                 _init_obj_list,
                 _max_depth,
                 _coll_mngr,
                 _meshes,
                 _contact_points,
                 _contact_faces,
                 _rotation_types,
                 _min_visit=1,
                 _goal_obj=None,
                 _physcial_constraint_checker=geometry_based_physical_checker,
                 _exploration=None,
                 _init_left_joint_values=None,
                 _init_left_gripper_width=0.03,
                 _init_right_joint_values=None,
                 _init_right_gripper_width=0.03,
                 _get_planning_scene_proxy=None,
                 _apply_planning_scene_proxy=None,
                 _compute_fk_proxy=None,
                 _cartesian_planning_with_gripper_pose_proxy=None,
                 _planning_with_gripper_pose_proxy=None,
                 _planning_with_arm_joints_proxy=None):

        if _init_left_joint_values is None:
            _init_left_joint_values = {'left_w0': 0.6699952259595108,
                                       'left_w1': 1.030009435085784,
                                       'left_w2': -0.4999997247485215,
                                       'left_e0': -1.189968899785275,
                                       'left_e1': 1.9400238130755056,
                                       'left_s0': -0.08000397926829805,
                                       'left_s1': -0.9999781166910306}
        if _init_right_joint_values is None:
            _init_right_joint_values = {'right_w0': -0.6699952259595108,
                                        'right_w1': 1.030009435085784,
                                        'right_w2': 0.4999997247485215,
                                        'right_e0': 1.189968899785275,
                                        'right_e1': 1.9400238130755056,
                                        'right_s0': -0.08000397926829805,
                                        'right_s1': -0.9999781166910306}

        self.Tree = nx.DiGraph()
        self.max_depth = _max_depth
        self.min_visit = _min_visit
        self.Tree.add_node(0)
        self.Tree.update(nodes=[(0, {'depth': 0,
                                     'state': _init_obj_list,
                                     'reward': 0.,
                                     'value': -np.inf,
                                     'visit': 0,
                                     'left_joint_values': _init_left_joint_values,
                                     'left_gripper_width': _init_left_gripper_width,
                                     'right_joint_values': _init_right_joint_values,
                                     'right_gripper_width': _init_right_gripper_width})])

        self.coll_mngr = _coll_mngr
        self.meshes = _meshes
        self.contact_points = _contact_points
        self.contact_faces = _contact_faces
        self.rotation_types = _rotation_types

        self.goal_obj = _goal_obj
        if _goal_obj is None:
            self.side_place_flag = True
        else:
            self.side_place_flag = False
        self.physcial_constraint_checker = _physcial_constraint_checker
        self.network = None

        if _exploration is None:
            _exploration = {'method': 'random'}
        self.exploration_method = _exploration

        self.init_left_joint_values = _init_left_joint_values
        self.init_left_gripper_width = _init_left_gripper_width
        self.init_right_joint_values = _init_right_joint_values
        self.init_right_gripper_width = _init_right_gripper_width
        self.get_planning_scene_proxy = _get_planning_scene_proxy
        self.apply_planning_scene_proxy = _apply_planning_scene_proxy
        self.compute_fk_proxy = _compute_fk_proxy
        self.cartesian_planning_with_gripper_pose_proxy = _cartesian_planning_with_gripper_pose_proxy
        self.planning_with_gripper_pose_proxy = _planning_with_gripper_pose_proxy
        self.planning_with_arm_joints_proxy = _planning_with_arm_joints_proxy

    def exploration(self, state_node):
        depth = self.Tree.nodes[state_node]['depth']
        visit = self.Tree.nodes[state_node]['visit']
        left_joint_values = self.Tree.nodes[state_node]['left_joint_values']
        left_gripper_width = self.Tree.nodes[state_node]['left_gripper_width']
        right_joint_values = self.Tree.nodes[state_node]['right_joint_values']
        right_gripper_width = self.Tree.nodes[state_node]['right_gripper_width']

        self.Tree.update(nodes=[(state_node, {'visit': visit + 1})])

        if depth <= self.max_depth:
            obj_list = self.Tree.nodes[state_node]['state']
            action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
            if obj_list is None:
                return 0.0
            elif len(action_nodes) == 0:
                action_list = get_possible_actions(obj_list, self.meshes, coll_mngr, contact_points, contact_faces, rotation_types,
                                                   side_place_flag=self.side_place_flag,
                                                   goal_obj=self.goal_obj)
                if len(action_list) == 0:
                    return 0.0
                else:
                    for action in action_list:
                        set_of_next_obj_list = get_possible_transitions(obj_list, action, _physical_checker=lambda x, y, z: self.physcial_constraint_checker(x, y, z, self.meshes, network=self.network))
                        if len(set_of_next_obj_list) > 0:
                            reward_list = []
                            for next_obj_list in set_of_next_obj_list:
                                reward = get_reward(obj_list, action, self.goal_obj, next_obj_list, self.meshes)
                                reward_list.append(reward)

                            sort_indices = np.argsort(reward_list)
                            reward_list = [reward_list[i] for i in sort_indices]
                            set_of_next_obj_list = [set_of_next_obj_list[i] for i in sort_indices]

                            child_action_node = self.Tree.number_of_nodes()
                            self.Tree.add_node(child_action_node)
                            self.Tree.update(nodes=[(child_action_node,
                                                     {'depth': depth,
                                                      'state': obj_list,
                                                      'action': action,
                                                      'reward': 0.,
                                                      'value': -np.inf,
                                                      'done': False,
                                                      'visit': 0,
                                                      'next_states': set_of_next_obj_list,
                                                      'next_rewards': reward_list,
                                                      'left_joint_values': left_joint_values,
                                                      'left_gripper_width': left_gripper_width,
                                                      'right_joint_values': right_joint_values,
                                                      'right_gripper_width': right_gripper_width})])
                            self.Tree.add_edge(state_node, child_action_node)
                    action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]

            if len(action_nodes) > 0:
                action_values = [self.Tree.nodes[action_node]['value'] for action_node in action_nodes]
                action_visits = [self.Tree.nodes[action_node]['visit'] for action_node in action_nodes]
                action_list = [self.Tree.nodes[action_node]['action'] for action_node in action_nodes]

                eps = np.maximum(np.minimum(1., 1 / np.maximum(visit, 1)), 0.01)
                if np.any(['place' in action['type'] for action in action_list]):
                    if self.goal_obj is not None:
                        table_place_indices = [action_idx for action_idx, action in enumerate(action_list) if
                                               'table' in action['param']]
                        if len(table_place_indices) > 0:
                            selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, _indices=table_place_indices, eps=eps)
                        else:
                            selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=eps)
                    else:
                        non_table_place_indices = [action_idx for action_idx, action in enumerate(action_list) if
                                                   'table' not in action['param']]
                        if len(non_table_place_indices) > 0:
                            selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, _indices=non_table_place_indices, eps=eps)
                        else:
                            selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=eps)
                else:
                    selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=eps)
                selected_action_node = action_nodes[selected_idx]
                selected_action_value = action_values[selected_idx]
                selected_action_value_new = self.action_exploration(selected_action_node)

                if selected_action_value < selected_action_value_new:
                    action_values[selected_idx] = selected_action_value_new
                    self.Tree.update(nodes=[(state_node, {'value': np.max(action_values)})])
                return np.max(action_values)
            else:
                return 0.0
        else:
            return 0.0

    def action_exploration(self, action_node):
        depth = self.Tree.nodes[action_node]['depth']
        action_value = self.Tree.nodes[action_node]['value']
        visit = self.Tree.nodes[action_node]['visit']
        left_joint_values = self.Tree.nodes[action_node]['left_joint_values']
        left_gripper_width = self.Tree.nodes[action_node]['left_gripper_width']
        right_joint_values = self.Tree.nodes[action_node]['right_joint_values']
        right_gripper_width = self.Tree.nodes[action_node]['right_gripper_width']

        self.Tree.update(nodes=[(action_node, {'visit': visit + 1})])

        next_states = self.Tree.nodes[action_node]['next_states']
        next_rewards = self.Tree.nodes[action_node]['next_rewards']
        next_state, next_reward = next_states[-1], next_rewards[-1]
        child_node = self.Tree.number_of_nodes()
        self.Tree.add_node(child_node)
        self.Tree.update(nodes=[(child_node,
                                 {'depth': depth + 1,
                                  'state': next_state,
                                  'reward': next_reward,
                                  'value': -np.inf,
                                  'visit': 0,
                                  'left_joint_values': left_joint_values,
                                  'left_gripper_width': left_gripper_width,
                                  'right_joint_values': right_joint_values,
                                  'right_gripper_width': right_gripper_width,
                                  'planned_traj_list': []})])
        self.Tree.add_edge(action_node, child_node)

        next_state_node = child_node

        action_value_new = next_reward + self.exploration(next_state_node)
        if action_value < action_value_new:
            self.Tree.update(nodes=[(action_node, {'value': action_value_new})])

        return action_value_new

    def kinematic_exploration_v3(self, state_node=0):
        depth = self.KinematicTree.nodes[state_node]['depth']
        visit = self.KinematicTree.nodes[state_node]['true_visit']
        state_value = self.KinematicTree.nodes[state_node]['true_value']
        obj_list = self.KinematicTree.nodes[state_node]['state']
        left_joint_values = self.KinematicTree.nodes[state_node]['left_joint_values']
        left_gripper_width = self.KinematicTree.nodes[state_node]['left_gripper_width']
        right_joint_values = self.KinematicTree.nodes[state_node]['right_joint_values']
        right_gripper_width = self.KinematicTree.nodes[state_node]['right_gripper_width']
        self.Tree.update(nodes=[(state_node, {'true_visit': visit + 1})])

        action_nodes = [action_node for action_node in self.KinematicTree.neighbors(state_node) if not np.isinf(self.KinematicTree.nodes[action_node]['true_reward'])]
        action_values = [self.KinematicTree.nodes[action_node]['true_value'] for action_node in action_nodes]
        action_visits = [self.KinematicTree.nodes[action_node]['true_visit'] for action_node in action_nodes]
        action_list = [self.KinematicTree.nodes[action_node]['action'] for action_node in action_nodes]
        print("============= Action node information ============")
        print(depth, "/", self.max_depth, "state node :", state_node)
        print(depth, "/", self.max_depth, "possible actions / total actions :", len(action_nodes), len([action_node for action_node in self.KinematicTree.neighbors(state_node)]))
        if len(action_nodes) > 0:
            selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=0.1)
            selected_action_node = action_nodes[selected_idx]
            selected_action_value = action_values[selected_idx]
            selected_action_visit = action_visits[selected_idx]
            selected_action = action_list[selected_idx]
            self.Tree.update(nodes=[(selected_action_node, {'true_visit': selected_action_visit + 1})])
            print("============= Selected action node information ============")
            print(depth, "/", self.max_depth, "selected action node :", selected_action_node)
            print(depth, "/", self.max_depth, selected_action['type'], selected_action['param'])

            next_state_nodes = [node for node in self.KinematicTree.neighbors(selected_action_node)]
            if len(next_state_nodes) > 0:
                next_state_node = next_state_nodes[0]
                next_visit = self.KinematicTree.nodes[next_state_node]['true_visit']
                next_object_list = self.KinematicTree.nodes[next_state_node]['state']
                self.Tree.update(nodes=[(next_state_node, {'true_visit': next_visit + 1})])
                kinematic_planning_flag = False
                new_next_object_list = None
                if not self.KinematicTree.nodes[selected_action_node]['done']:
                    new_next_object_list, next_left_joint_values, next_left_gripper_width, next_right_joint_values, next_right_gripper_width, planned_traj_list = \
                        kinematic_planning(obj_list, next_object_list,
                                           left_joint_values, left_gripper_width,
                                           right_joint_values, right_gripper_width,
                                           selected_action, self.meshes,
                                           _get_planning_scene_proxy=self.get_planning_scene_proxy,
                                           _apply_planning_scene_proxy=self.apply_planning_scene_proxy,
                                           _cartesian_planning_with_gripper_pose_proxy=self.cartesian_planning_with_gripper_pose_proxy,
                                           _planning_with_gripper_pose_proxy=self.planning_with_gripper_pose_proxy,
                                           _planning_with_arm_joints_proxy=self.planning_with_arm_joints_proxy,
                                           _compute_fk_proxy=self.compute_fk_proxy)
                    if new_next_object_list is not None:
                        kinematic_planning_flag = True
                else:
                    kinematic_planning_flag = True

                if kinematic_planning_flag:
                    if new_next_object_list is not None:
                        self.Tree.update(nodes=[(next_state_node, {'left_joint_values': next_left_joint_values})])
                        self.Tree.update(nodes=[(next_state_node, {'left_gripper_width': next_left_gripper_width})])
                        self.Tree.update(nodes=[(next_state_node, {'right_joint_values': next_right_joint_values})])
                        self.Tree.update(nodes=[(next_state_node, {'right_gripper_width': next_right_gripper_width})])
                        self.Tree.update(nodes=[(next_state_node, {'planned_traj_list': planned_traj_list})])
                        self.Tree.update(nodes=[(selected_action_node, {'done': True})])

                    new_action_value = self.KinematicTree.nodes[next_state_node]['true_reward'] + self.kinematic_exploration_v3(next_state_node)
                    if selected_action_value < new_action_value:
                        selected_action_value = new_action_value
                        self.Tree.update(nodes=[(selected_action_node, {'true_value': selected_action_value})])

                    if state_value < new_action_value:
                        state_value = new_action_value
                        self.Tree.update(nodes=[(state_node, {'true_value': state_value})])
                    return state_value
                else:
                    print("planning fails")
                    self.Tree.update(nodes=[(selected_action_node, {'true_reward': -np.inf})])
                    self.Tree.update(nodes=[(selected_action_node, {'true_value': -np.inf})])
                    return -np.inf
            else:
                print("there is no next state node")
                self.Tree.update(nodes=[(selected_action_node, {'true_value': 0.0})])
                return 0.0
        else:
            return 0.

    def exhaustive_kinematic_search(self, state_node=0):
        obj_list = self.Tree.nodes[state_node]['state']
        left_joint_values = self.Tree.nodes[state_node]['left_joint_values']
        left_gripper_width = self.Tree.nodes[state_node]['left_gripper_width']
        right_joint_values = self.Tree.nodes[state_node]['right_joint_values']
        right_gripper_width = self.Tree.nodes[state_node]['right_gripper_width']

        action_values = [self.Tree.nodes[action_node]['value'] for action_node in self.Tree.neighbors(state_node)]
        action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
        for i in np.argsort(action_values):
            action_node = action_nodes[i]
            action = self.Tree.nodes[action_node]['action']
            next_state_nodes = [node for node in self.Tree.neighbors(action_node)]
            for next_state_node in next_state_nodes:
                next_obj_list = self.Tree.nodes[next_state_node]['state']
                new_next_object_list, next_left_joint_values, next_left_gripper_width, next_right_joint_values, next_right_gripper_width, planned_traj_list = \
                    kinematic_planning(obj_list, next_obj_list,
                                       left_joint_values, left_gripper_width,
                                       right_joint_values, right_gripper_width,
                                       action, self.meshes,
                                       _get_planning_scene_proxy=self.get_planning_scene_proxy,
                                       _apply_planning_scene_proxy=self.apply_planning_scene_proxy,
                                       _cartesian_planning_with_gripper_pose_proxy=self.cartesian_planning_with_gripper_pose_proxy,
                                       _planning_with_gripper_pose_proxy=self.planning_with_gripper_pose_proxy,
                                       _planning_with_arm_joints_proxy=self.planning_with_arm_joints_proxy,
                                       _compute_fk_proxy=self.compute_fk_proxy)
                visualize(new_next_object_list, self.meshes, self.goal_obj)

                if new_next_object_list is not None:
                    self.Tree.update(nodes=[(next_state_node, {'state': new_next_object_list})])
                    self.Tree.update(nodes=[(next_state_node, {'left_joint_values': next_left_joint_values})])
                    self.Tree.update(nodes=[(next_state_node, {'left_gripper_width': next_left_gripper_width})])
                    self.Tree.update(nodes=[(next_state_node, {'right_joint_values': next_right_joint_values})])
                    self.Tree.update(nodes=[(next_state_node, {'right_gripper_width': next_right_gripper_width})])
                    self.Tree.update(nodes=[(next_state_node, {'planned_traj_list': planned_traj_list})])
                    self.exhaustive_kinematic_search(next_state_node)

    def exhaustive_search(self, state_node):
        depth = self.Tree.nodes[state_node]['depth']
        visit = self.Tree.nodes[state_node]['visit']
        left_joint_values = self.Tree.nodes[state_node]['left_joint_values']
        left_gripper_width = self.Tree.nodes[state_node]['left_gripper_width']
        right_joint_values = self.Tree.nodes[state_node]['right_joint_values']
        right_gripper_width = self.Tree.nodes[state_node]['right_gripper_width']
        self.Tree.update(nodes=[(state_node, {'visit': visit + 1})])

        if depth <= self.max_depth:
            obj_list = self.Tree.nodes[state_node]['state']
            action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
            if obj_list is None:
                return 0.0
            elif len(action_nodes) == 0:
                action_list = get_possible_actions(obj_list, self.meshes, self.coll_mngr, self.contact_points, self.contact_faces, self.rotation_types,
                                                   side_place_flag=self.side_place_flag,
                                                   goal_obj=self.goal_obj)
                if len(action_list) == 0:
                    return 0.0
                else:
                    for action in action_list:
                        set_of_next_obj_list = get_possible_transitions(obj_list, action, _physical_checker=lambda x, y, z: self.physcial_constraint_checker(x, y, z, self.meshes, network=self.network))
                        if len(set_of_next_obj_list) > 0:
                            reward_list = []
                            for next_obj_list in set_of_next_obj_list:
                                reward = get_reward(obj_list, action, self.goal_obj, next_obj_list, self.meshes)
                                reward_list.append(reward)

                            sort_indices = np.argsort(reward_list)
                            reward_list = [reward_list[i] for i in sort_indices]
                            set_of_next_obj_list = [set_of_next_obj_list[i] for i in sort_indices]

                            if action["type"] is "pick":
                                action["grasp_poses"] = [action["grasp_poses"][i] for i in sort_indices]
                                action["retreat_poses"] = [action["retreat_poses"][i] for i in sort_indices]
                                action["gripper_widths"] = [action["gripper_widths"][i] for i in sort_indices]
                            else:
                                action["placing_poses"] = [action["placing_poses"][i] for i in sort_indices]
                            print(sort_indices)

                            child_action_node = self.Tree.number_of_nodes()
                            self.Tree.add_node(child_action_node)
                            self.Tree.update(nodes=[(child_action_node,
                                                     {'depth': depth,
                                                      'state': obj_list,
                                                      'action': action,
                                                      'reward': 0.,
                                                      'value': -np.inf,
                                                      'done': False,
                                                      'visit': 0,
                                                      'next_states': set_of_next_obj_list,
                                                      'next_rewards': reward_list,
                                                      'left_joint_values': left_joint_values,
                                                      'left_gripper_width': left_gripper_width,
                                                      'right_joint_values': right_joint_values,
                                                      'right_gripper_width': right_gripper_width})])
                            self.Tree.add_edge(state_node, child_action_node)
                    action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]

            if len(action_nodes) > 0:
                for action_node in action_nodes:
                    self.action_exhaustive_search(action_node)

                action_values = [self.Tree.nodes[action_node]['value'] for action_node in action_nodes]
                self.Tree.update(nodes=[(state_node, {'value': np.max(action_values)})])
                return np.max(action_values)
            else:
                return 0.0
        else:
            return 0.0

    def action_exhaustive_search(self, action_node):
        depth = self.Tree.nodes[action_node]['depth']
        action = self.Tree.nodes[action_node]['action']
        visit = self.Tree.nodes[action_node]['visit']
        left_joint_values = self.Tree.nodes[action_node]['left_joint_values']
        left_gripper_width = self.Tree.nodes[action_node]['left_gripper_width']
        right_joint_values = self.Tree.nodes[action_node]['right_joint_values']
        right_gripper_width = self.Tree.nodes[action_node]['right_gripper_width']

        self.Tree.update(nodes=[(action_node, {'visit': visit + 1})])

        next_states = self.Tree.nodes[action_node]['next_states']
        next_rewards = self.Tree.nodes[action_node]['next_rewards']

        if action["type"] is "pick":
            # for next_state, next_reward in zip(next_states, next_rewards):
            next_state, next_reward = next_states[-1], next_rewards[-1]
            child_node = self.Tree.number_of_nodes()
            self.Tree.add_node(child_node)
            self.Tree.update(nodes=[(child_node,
                                     {'depth': depth + 1,
                                      'state': next_state,
                                      'reward': next_reward,
                                      'value': -np.inf,
                                      'visit': 0,
                                      'left_joint_values': left_joint_values,
                                      'left_gripper_width': left_gripper_width,
                                      'right_joint_values': right_joint_values,
                                      'right_gripper_width': right_gripper_width,
                                      'planned_traj_list': []})])
            self.Tree.add_edge(action_node, child_node)

            next_state_node = child_node
            value = next_reward + self.exhaustive_search(next_state_node)
            # value = np.max([self.Tree.nodes[next_state_node]['reward'] + self.exhaustive_search(next_state_node) for next_state_node in self.Tree.neighbors(action_node)])
        else:
            next_state, next_reward = next_states[-1], next_rewards[-1]
            child_node = self.Tree.number_of_nodes()
            self.Tree.add_node(child_node)
            self.Tree.update(nodes=[(child_node,
                                     {'depth': depth + 1,
                                      'state': next_state,
                                      'reward': next_reward,
                                      'value': -np.inf,
                                      'visit': 0,
                                      'left_joint_values': left_joint_values,
                                      'left_gripper_width': left_gripper_width,
                                      'right_joint_values': right_joint_values,
                                      'right_gripper_width': right_gripper_width,
                                      'planned_traj_list': []})])
            self.Tree.add_edge(action_node, child_node)

            next_state_node = child_node
            value = next_reward + self.exhaustive_search(next_state_node)

        self.Tree.update(nodes=[(action_node, {'value': value})])
        print(depth, "/", self.max_depth, " is finished")
        return value

    def get_best_path(self, start_node=0):
        next_nodes = [next_node for next_node in self.Tree.neighbors(start_node)]
        if len(next_nodes) == 0:
            return [start_node]
        else:
            best_idx = np.argmax([self.Tree.nodes[next_node]['value'] for next_node in next_nodes])
            next_node = next_nodes[best_idx]
            return [start_node, ] + self.get_best_path(next_node)

    def get_all_kinematic_path(self, state_node=0):
        paths = []
        values = []
        kinematic_plans = []

        action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
        if len(action_nodes) > 0:
            for action_node in action_nodes:
                next_state_nodes = [node for node in self.Tree.neighbors(action_node)]
                for next_state_node in next_state_nodes:
                    if len(self.Tree.nodes[next_state_node]['planned_traj_list']) > 0:
                        reward = self.Tree.nodes[next_state_node]['reward']
                        sub_tree_paths, sub_tree_values, sub_tree_kinematic_plans = self.get_all_kinematic_path(next_state_node)
                        for sub_tree_path, sub_tree_value, sub_tree_kinematic_plan in zip(sub_tree_paths, sub_tree_values, sub_tree_kinematic_plans):
                            paths.append([state_node, action_node, next_state_node]+sub_tree_path)
                            values.append(reward + sub_tree_value)
                            kinematic_plans.append(self.Tree.nodes[next_state_node]['planned_traj_list'] + sub_tree_kinematic_plan)
            path_indices = np.argsort(values)
            print(path_indices)
            values = values[path_indices]
            paths = paths[path_indices]
            kinematic_plans = kinematic_plans[path_indices]
        return paths, values, kinematic_plans

    def visualize(self):
        visited_nodes = [n for n in self.Tree.nodes if self.Tree.nodes[n]['visit'] > 0]
        visited_tree = self.Tree.subgraph(visited_nodes)
        labels = {
            n: 'depth:{:d}\nvisit:{:d}\nreward:{:.4f}\nvalue:{:.4f}'.format(visited_tree.nodes[n]['depth'], visited_tree.nodes[n]['visit'], visited_tree.nodes[n]['reward'], visited_tree.nodes[n]['value'])
            for n in visited_tree.nodes}

        # nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        # same layout using matplotlib with no labels
        # plt.figure(figsize=(32, 64))
        plt.figure()
        plt.title('')
        # nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        pos = graphviz_layout(visited_tree, prog='dot')
        nx.draw(visited_tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        plt.show()


if __name__ == '__main__':
    # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=.0015)
    # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
    #     configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
    #                               goal_name='tower_goal')

    # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=1.)
    # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
    #     configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
    #                               goal_name='stack_easy')

    # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=0.003)
    # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
    #     configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
    #                               goal_name='box_goal')

    mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=1.)
    initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
        configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
                                  goal_name='stack_very_easy')

    mcts = Tree(initial_object_list, np.sum(n_obj_per_mesh_types) * 2, coll_mngr, meshes, contact_points,
                contact_faces, rotation_types, _goal_obj=goal_obj,
                _exploration={'method': 'random', 'param': 1.})
    opt_num = 30
    best_value = -np.inf
    opt_idx = 0

    # while opt_idx < opt_num:
    #     mcts.exploration(0)
    #     if best_value < mcts.Tree.nodes[0]['value']:
    #         best_value = mcts.Tree.nodes[0]['value']
    #         print(opt_idx, best_value)
    #     if (opt_idx + 1) % 5 == 0:
    #         print('============={}/{}============='.format(opt_idx, opt_num))
    #     opt_idx += 1

    mcts.exhaustive_search(state_node=0)
    mcts.visualize()
    print(mcts.Tree.nodes[0]['value'])
    best_path_indices = mcts.get_best_path(0)
    object_list = mcts.Tree.nodes[best_path_indices[-1]]['state']
    visualize(object_list, meshes, _goal_obj=goal_obj)
