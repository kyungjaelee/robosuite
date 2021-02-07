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

from robosuite.mcts.util import *
# from robosuite.utils import transform_utils as tf
# import trimesh.proximity

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# import torch

from matplotlib import pyplot as plt


#
#
# def transform_matrix2pose(T):
#     q = tf.mat2quat(T)
#     orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
#     position = Point(x=T[0, 3], y=T[1, 3], z=T[2, 3] - 0.93)
#     pose = Pose(position, orientation)
#
#     return pose
#
#
# def pose2transform_matrix(pose):
#     orientation = pose.orientation
#     position = pose.position
#
#     q = [orientation.x, orientation.y, orientation.z, orientation.w]
#     T = np.eye(4)
#     T[:3, :3] = tf.quat2mat(q)
#     T[0, 3] = position.x
#     T[1, 3] = position.y
#     T[2, 3] = position.z
#
#     return T
#
#
# def synchronize_planning_scene(left_joint_values,
#                                right_joint_values,
#                                object_list,
#                                meshes,
#                                left_gripper_width=0.0,
#                                right_gripper_width=0.0,
#                                get_planning_scene_proxy=None,
#                                apply_planning_scene_proxy=None):
#     joint_values = {}
#     joint_values.update(left_joint_values)
#     joint_values.update(right_joint_values)
#
#     resp = get_planning_scene_proxy(GetPlanningSceneRequest())
#     current_scene = resp.scene
#
#     next_scene = deepcopy(current_scene)
#     next_scene.robot_state.joint_state.name = list(joint_values.keys())
#     next_scene.robot_state.joint_state.position = list(joint_values.values())
#     next_scene.robot_state.joint_state.velocity = [0] * len(joint_values)
#     next_scene.robot_state.joint_state.effort = [0] * len(joint_values)
#     next_scene.robot_state.is_diff = True
#
#     held_obj_idx = get_held_object(object_list)
#     if held_obj_idx is None and len(next_scene.robot_state.attached_collision_objects) > 0:
#         for attached_object_idx in range(len(next_scene.robot_state.attached_collision_objects)):
#             next_scene.robot_state.attached_collision_objects[
#                 attached_object_idx].object.operation = CollisionObject.REMOVE
#     if len(next_scene.world.collision_objects) > 0:
#         for scene_object_idx in range(len(next_scene.world.collision_objects)):
#             next_scene.world.collision_objects[scene_object_idx].operation = CollisionObject.REMOVE
#
#     for obj in object_list:
#         co = CollisionObject()
#         co.operation = CollisionObject.ADD
#         co.id = obj.name
#         co.header = next_scene.robot_state.joint_state.header
#
#         mesh = Mesh()
#         for face in meshes[obj.mesh_idx].faces:
#             triangle = MeshTriangle()
#             triangle.vertex_indices = face
#             mesh.triangles.append(triangle)
#
#         for vertex in meshes[obj.mesh_idx].vertices:
#             point = Point()
#             point.x = vertex[0]
#             point.y = vertex[1]
#             point.z = vertex[2]
#             mesh.vertices.append(point)
#
#         co.meshes = [mesh]
#
#         obj_pose = transform_matrix2pose(obj.pose)
#         co.mesh_poses = [obj_pose]
#
#         if held_obj_idx is not None and len(next_scene.robot_state.attached_collision_objects) == 0 \
#                 and object_list[held_obj_idx].name is obj.name:
#             aco = AttachedCollisionObject()
#             aco.link_name = 'left_gripper'
#             aco.object = co
#             next_scene.robot_state.attached_collision_objects.append(aco)
#         else:
#             next_scene.world.collision_objects.append(co)
#
#     next_scene.robot_state.joint_state.name += ('l_gripper_l_finger_joint', 'l_gripper_r_finger_joint')
#     next_scene.robot_state.joint_state.position += (np.minimum(0.02, left_gripper_width), -np.minimum(0.02, left_gripper_width))
#
#     next_scene.robot_state.joint_state.name += ('r_gripper_l_finger_joint', 'r_gripper_r_finger_joint')
#     next_scene.robot_state.joint_state.position += (
#     np.minimum(0.02, right_gripper_width), -np.minimum(0.02, right_gripper_width))
#
#     next_scene.is_diff = True
#     req = ApplyPlanningSceneRequest()
#     req.scene = next_scene
#     resp = apply_planning_scene_proxy(req)
#     for _ in range(100):
#         rospy.sleep(0.001)


# def get_transition_with_baxter(left_arm_joint_values,
#                                right_arm_joint_values,
#                                left_arm_init_joint_values,
#                                right_arm_init_joint_values,
#                                object_list,
#                                left_gripper_width,
#                                right_gripper_width,
#                                action,
#                                meshes,
#                                coll_mngr,
#                                goal_obj=None,
#                                get_planning_scene_proxy=None,
#                                apply_planning_scene_proxy=None,
#                                compute_fk_proxy=None,
#                                mujoco_moveit_planner_proxy=None,
#                                n_planning_trial=10):
#     for obj in object_list:
#         coll_mngr.set_transform(obj.name, obj.pose)
#
#     synchronize_planning_scene(left_arm_joint_values,
#                                right_arm_joint_values,
#                                object_list,
#                                meshes,
#                                left_gripper_width=left_gripper_width,
#                                right_gripper_width=right_gripper_width,
#                                get_planning_scene_proxy=get_planning_scene_proxy,
#                                apply_planning_scene_proxy=apply_planning_scene_proxy)
#
#     planning_results = []
#     planned_trajs = []
#     planned_gripper_widths = []
#
#     next_object_list = deepcopy(object_list)
#     next_left_arm_joint_values = left_arm_joint_values
#     next_right_arm_joint_values = right_arm_joint_values
#     next_left_gripper_width = left_gripper_width
#     next_right_gripper_width = right_gripper_width
#     if action["type"] is "pick":
#         pick_obj_idx = get_obj_idx_by_name(next_object_list, action['param'])
#         for _ in range(n_planning_trial):
#             next_gripper_pose, next_retreat_pose, _, next_gripper_width = sample_grasp_pose(
#                 next_object_list[pick_obj_idx], meshes)
#             if next_gripper_pose is None: continue
#
#             synchronize_planning_scene(left_arm_joint_values,
#                                        right_arm_joint_values,
#                                        next_object_list,
#                                        meshes,
#                                        left_gripper_width=next_gripper_width,
#                                        right_gripper_width=right_gripper_width,
#                                        get_planning_scene_proxy=get_planning_scene_proxy,
#                                        apply_planning_scene_proxy=apply_planning_scene_proxy)
#             moveit_req = MuJoCoMoveitConnectorRequest()
#             moveit_req.planning_type = "gripper_pose"
#             moveit_req.group_name = "left_arm"
#             moveit_req.target_gripper_pose = transform_matrix2pose(next_retreat_pose)
#             resp = mujoco_moveit_planner_proxy(moveit_req)
#             approach_planning_result, approach_planned_traj = resp.success, resp.joint_trajectory
#             if approach_planning_result:
#                 approach_left_arm_joint_values = {}
#                 for key, val in zip(approach_planned_traj.joint_names,
#                                     approach_planned_traj.points[-1].positions):
#                     approach_left_arm_joint_values[key] = val
#                 synchronize_planning_scene(approach_left_arm_joint_values,
#                                            next_right_arm_joint_values,
#                                            next_object_list,
#                                            meshes,
#                                            left_gripper_width=next_left_gripper_width,
#                                            right_gripper_width=next_right_gripper_width,
#                                            get_planning_scene_proxy=get_planning_scene_proxy,
#                                            apply_planning_scene_proxy=apply_planning_scene_proxy)
#
#                 moveit_req = MuJoCoMoveitConnectorRequest()
#                 moveit_req.planning_type = "cartesian"
#                 moveit_req.group_name = "left_arm"
#                 moveit_req.target_gripper_pose = transform_matrix2pose(next_gripper_pose)
#                 resp = mujoco_moveit_planner_proxy(moveit_req)
#                 planning_result, planned_traj = resp.success, resp.joint_trajectory
#                 if planning_result:
#                     next_left_arm_joint_values = {}
#                     for key, val in zip(planned_traj.joint_names,
#                                         planned_traj.points[-1].positions):
#                         next_left_arm_joint_values[key] = val
#                     next_left_gripper_width = next_gripper_width
#
#                     next_right_arm_joint_values = right_arm_joint_values
#                     next_right_gripper_width = right_gripper_width
#
#                     next_object_list[pick_obj_idx].logical_state.clear()
#                     next_object_list[pick_obj_idx].logical_state["held"] = []
#                     update_logical_state(next_object_list)
#
#                     synchronize_planning_scene(next_left_arm_joint_values,
#                                                next_right_arm_joint_values,
#                                                next_object_list,
#                                                meshes,
#                                                left_gripper_width=next_left_gripper_width,
#                                                right_gripper_width=next_right_gripper_width,
#                                                get_planning_scene_proxy=get_planning_scene_proxy,
#                                                apply_planning_scene_proxy=apply_planning_scene_proxy)
#
#                     planning_results.append(approach_planning_result)
#                     planned_trajs.append(approach_planned_traj)
#
#                     planning_results.append(planning_result)
#                     planned_trajs.append(planned_traj)
#                     planned_gripper_widths.append(next_left_gripper_width)
#                     return next_object_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths
#
#     if action["type"] is "place":
#         moveit_req = MuJoCoMoveitConnectorRequest()
#         moveit_req.planning_type = "joint_values"
#         moveit_req.group_name = "left_arm"
#         moveit_req.target_joint_names = left_arm_init_joint_values.keys()
#         moveit_req.target_joint_values = left_arm_init_joint_values.values()
#         resp = mujoco_moveit_planner_proxy(moveit_req)
#         pre_planning_result, pre_planned_traj = resp.success, resp.joint_trajectory
#         if pre_planning_result:
#             synchronize_planning_scene(left_arm_init_joint_values,
#                                        right_arm_joint_values,
#                                        next_object_list,
#                                        meshes,
#                                        left_gripper_width=next_left_gripper_width,
#                                        right_gripper_width=right_gripper_width,
#                                        get_planning_scene_proxy=get_planning_scene_proxy,
#                                        apply_planning_scene_proxy=apply_planning_scene_proxy)
#
#             planning_results.append(pre_planning_result)
#             planned_trajs.append(pre_planned_traj)
#             planned_gripper_widths.append(left_gripper_width)
#
#             held_obj_idx = get_held_object(next_object_list)
#             resp = get_planning_scene_proxy(GetPlanningSceneRequest())
#             current_scene = resp.scene
#             rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
#             rel_T = pose2transform_matrix(rel_obj_pose)
#
#             req = GetPositionFKRequest()
#             req.fk_link_names = ['left_gripper']
#             req.header.frame_id = 'world'
#             req.robot_state = current_scene.robot_state
#             resp = compute_fk_proxy(req)
#
#             gripper_T = pose2transform_matrix(resp.pose_stamped[0].pose)
#             next_object_pose = gripper_T.dot(rel_T)
#             next_object_pose[2, 3] += 0.93
#
#             next_object_list[held_obj_idx].pose = next_object_pose
#
#             place_obj_idx = get_obj_idx_by_name(next_object_list, action['param'])
#             if "on" in next_object_list[held_obj_idx].logical_state:
#                 support_obj_idx = get_obj_idx_by_name(next_object_list,
#                                                       next_object_list[held_obj_idx].logical_state["on"][0])
#                 next_object_list[support_obj_idx].logical_state["support"].remove(next_object_list[held_obj_idx].name)
#                 next_object_list[held_obj_idx].logical_state.pop("on")
#
#             for _ in range(n_planning_trial):
#                 next_object_place_pose = sample_on_pose(next_object_list[place_obj_idx], next_object_list[held_obj_idx],
#                                                         meshes, coll_mngr)
#                 if next_object_place_pose is None:
#                     continue
#
#                 resp = get_planning_scene_proxy(GetPlanningSceneRequest())
#                 current_scene = resp.scene
#                 rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
#                 rel_T = pose2transform_matrix(rel_obj_pose)
#
#                 next_gripper_pose = next_object_place_pose.dot(np.linalg.inv(rel_T))
#                 moveit_req = MuJoCoMoveitConnectorRequest()
#                 moveit_req.planning_type = "gripper_pose"
#                 moveit_req.group_name = "left_arm"
#                 moveit_req.target_gripper_pose = transform_matrix2pose(next_gripper_pose)
#                 resp = mujoco_moveit_planner_proxy(moveit_req)
#                 planning_result, planned_traj = resp.success, resp.joint_trajectory
#                 if planning_result:
#                     next_left_arm_joint_values = {}
#                     for key, val in zip(planned_traj.joint_names,
#                                         planned_traj.points[-1].positions):
#                         next_left_arm_joint_values[key] = val
#                     next_left_gripper_width = left_gripper_width
#
#                     next_right_arm_joint_values = right_arm_joint_values
#                     next_right_gripper_width = right_gripper_width
#
#                     next_object_list[held_obj_idx].pose = next_object_place_pose
#                     next_object_list[held_obj_idx].logical_state.clear()
#                     next_object_list[held_obj_idx].logical_state["on"] = [next_object_list[place_obj_idx].name]
#                     update_logical_state(next_object_list)
#
#                     synchronize_planning_scene(next_left_arm_joint_values,
#                                                next_right_arm_joint_values,
#                                                next_object_list,
#                                                meshes,
#                                                left_gripper_width=next_left_gripper_width,
#                                                right_gripper_width=next_right_gripper_width,
#                                                get_planning_scene_proxy=get_planning_scene_proxy,
#                                                apply_planning_scene_proxy=apply_planning_scene_proxy)
#                     planning_results.append(planning_result)
#                     planned_trajs.append(planned_traj)
#                     planned_gripper_widths.append(next_left_gripper_width)
#
#                     moveit_req = MuJoCoMoveitConnectorRequest()
#                     moveit_req.planning_type = "joint_values"
#                     moveit_req.group_name = "left_arm"
#                     moveit_req.target_joint_names = left_arm_init_joint_values.keys()
#                     moveit_req.target_joint_values = left_arm_init_joint_values.values()
#                     resp = mujoco_moveit_planner_proxy(moveit_req)
#                     after_planning_result, after_planned_traj = resp.success, resp.joint_trajectory
#                     if after_planning_result:
#                         next_left_arm_joint_values = {}
#                         for key, val in zip(after_planned_traj.joint_names,
#                                             after_planned_traj.points[-1].positions):
#                             next_left_arm_joint_values[key] = val
#                         next_left_gripper_width = left_gripper_width
#
#                         next_right_arm_joint_values = right_arm_joint_values
#                         next_right_gripper_width = right_gripper_width
#
#                         synchronize_planning_scene(next_left_arm_joint_values,
#                                                    next_right_arm_joint_values,
#                                                    next_object_list,
#                                                    meshes,
#                                                    left_gripper_width=next_left_gripper_width,
#                                                    right_gripper_width=next_right_gripper_width,
#                                                    get_planning_scene_proxy=get_planning_scene_proxy,
#                                                    apply_planning_scene_proxy=apply_planning_scene_proxy)
#                         planning_results.append(after_planning_result)
#                         planned_trajs.append(after_planned_traj)
#                         planned_gripper_widths.append(next_left_gripper_width)
#                         return next_object_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths
#
#     planning_results = []
#     planned_trajs = []
#     planned_gripper_widths = []
#
#     next_object_list = deepcopy(object_list)
#     next_left_arm_joint_values = left_arm_joint_values
#     next_right_arm_joint_values = right_arm_joint_values
#     next_left_gripper_width = left_gripper_width
#     next_right_gripper_width = right_gripper_width
#     return next_object_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths


def check_stability(obj_idx, object_list, meshes, mesh1):
    if "on" in object_list[obj_idx].logical_state:
        obj_name = object_list[obj_idx].logical_state["on"][0]
        support_obj_idx = get_obj_idx_by_name(object_list, obj_name)
        mesh2 = deepcopy(meshes[object_list[support_obj_idx].mesh_idx])
        mesh2.apply_transform(object_list[support_obj_idx].pose)

        closest_points, dists, surface_idx = trimesh.proximity.closest_point(mesh2, [mesh1.center_mass])
        project_point2 = mesh1.center_mass - closest_points[0]
        project_point2 = project_point2 / np.sqrt(np.sum(project_point2 ** 2.))
        safe_com = project_point2[2] > 0.999 and mesh2.face_normals[surface_idx[0]][2] > 0.999

        if safe_com:
            mesh1 = trimesh.util.concatenate([mesh1, mesh2])
            return check_stability(support_obj_idx, object_list, meshes, mesh1)
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
        flag_ = check_stability(pick_idx, next_obj_list, meshes, mesh1)
        return flag_


#
#
# def mujoco_based_physical_checker(obj_list, action, next_obj_list, meshes, network=None):
#     if action['type'] is 'pick':
#         return True
#     if action['type'] is 'place':
#         return True
#
#
# def learning_based_physical_checker(obj_list, action, next_obj_list, meshes, network=None):
#     if action['type'] is 'pick':
#         return True
#     if action['type'] is 'place':
#         _, depths, masks = get_image(obj_list, action, next_obj_list, meshes=meshes)
#         images = torch.from_numpy(np.asarray([depths])).to('cuda')
#
#         # Forward pass
#         outputs = network(images)
#         _, predicted = torch.max(outputs.data, 1)
#         if predicted == 0:
#             return False
#         else:
#             return True


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
        # print(upper_bounds)
        # print(lower_bounds)
        # print(B_k)
        # print(u,b)
        # print("===================================")
    elif _exploration_method['method'] is 'bai_ucb':
        if len(_visits) == 1:
            selected_idx = 0
        else:
            # print(selected_action_values)
            # print(selected_visits)
            c = _exploration_method['param'] / np.maximum(_depth, 1)
            upper_bounds = selected_action_values + c * np.sqrt(1. / np.maximum(1., selected_visits))
            lower_bounds = selected_action_values - c * np.sqrt(1. / np.maximum(1., selected_visits))
            B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]
            b = np.argmin(B_k)
            u = np.argmax(upper_bounds)
            # print(upper_bounds)
            # print(lower_bounds)
            # print(B_k)
            # print(u,b)
            # print("===================================")
            if selected_visits[b] > selected_visits[u]:
                selected_idx = u
            else:
                selected_idx = b
    elif _exploration_method['method'] is 'bai_perturb':
        if len(_visits) == 1:
            selected_idx = 0
        else:
            # print(selected_action_values)
            # print(selected_visits)
            c = _exploration_method['param'] / np.maximum(_depth, 1)
            g = np.random.normal(size=(len(selected_visits)))
            upper_bounds = selected_action_values + c * np.sqrt(1. / np.maximum(1., selected_visits)) * g
            lower_bounds = selected_action_values - c * np.sqrt(1. / np.maximum(1., selected_visits)) * g
            B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]
            b = np.argmin(B_k)
            u = np.argmax(upper_bounds)
            # print(upper_bounds)
            # print(lower_bounds)
            # print(B_k)
            # print(u,b)
            # print("===================================")
            if selected_visits[b] > selected_visits[u]:
                selected_idx = u
            else:
                selected_idx = b
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
                 _with_robot=False,
                 _init_left_joint_values=None,
                 _init_left_gripper_width=0.03,
                 _init_right_joint_values=None,
                 _init_right_gripper_width=0.03,
                 _get_planning_scene_proxy=None,
                 _apply_planning_scene_proxy=None,
                 _compute_fk_proxy=None,
                 _planning_with_gripper_pose_proxy=None,
                 _planning_with_arm_joints_proxy=None):

        if _with_robot and _init_left_joint_values is None:
            _init_left_joint_values = {'left_w0': 0.6699952259595108,
                                       'left_w1': 1.030009435085784,
                                       'left_w2': -0.4999997247485215,
                                       'left_e0': -1.189968899785275,
                                       'left_e1': 1.9400238130755056,
                                       'left_s0': -0.08000397926829805,
                                       'left_s1': -0.9999781166910306}
        if _with_robot and _init_right_joint_values is None:
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
        if _with_robot:
            self.Tree.update(nodes=[(0, {'depth': 0,
                                         'state': _init_obj_list,
                                         'left_joint_values': _init_left_joint_values,
                                         'left_gripper_width': _init_left_gripper_width,
                                         'right_joint_values': _init_right_joint_values,
                                         'right_gripper_width': _init_right_gripper_width,
                                         'reward': 0,
                                         'value': -np.inf,
                                         'visit': 0})])
        else:
            self.Tree.update(nodes=[(0, {'depth': 0,
                                         'state': _init_obj_list,
                                         'reward': 0,
                                         'value': -np.inf,
                                         'visit': 0})])
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

        self.with_robot = _with_robot
        if self.with_robot:
            self.init_left_joint_values = _init_left_joint_values
            self.init_left_gripper_width = _init_left_gripper_width
            self.init_right_joint_values = _init_right_joint_values
            self.init_right_gripper_width = _init_right_gripper_width
            self.get_planning_scene_proxy = _get_planning_scene_proxy
            self.apply_planning_scene_proxy = _apply_planning_scene_proxy
            self.compute_fk_proxy = _compute_fk_proxy
            self.planning_with_gripper_pose_proxy = _planning_with_gripper_pose_proxy
            self.planning_with_arm_joints_proxy = _planning_with_arm_joints_proxy

    def exploration(self, state_node):
        depth = self.Tree.nodes[state_node]['depth']
        visit = self.Tree.nodes[state_node]['visit']
        if self.with_robot:
            left_joint_values = self.Tree.nodes[state_node]['left_joint_values']
            left_gripper_width = self.Tree.nodes[state_node]['left_gripper_width']
            right_joint_values = self.Tree.nodes[state_node]['right_joint_values']
            right_gripper_width = self.Tree.nodes[state_node]['right_gripper_width']
        self.Tree.update(nodes=[(state_node, {'visit': visit + 1})])

        if depth < self.max_depth:
            obj_list = self.Tree.nodes[state_node]['state']
            action_nodes = [action_node for action_node in self.Tree.neighbors(state_node) if self.Tree.nodes[action_node]['reward'] == 0.]
            if obj_list is None:
                return 0.0
            elif len(action_nodes) == 0:
                if self.with_robot:
                    action_list = get_possible_actions_with_robot(obj_list, self.meshes, self.coll_mngr,
                                                                  self.contact_points, self.contact_faces, self.rotation_types, side_place_flag=self.side_place_flag)
                else:
                    action_list = get_possible_actions(obj_list, self.meshes, self.coll_mngr,
                                                       self.contact_points, self.contact_faces, self.rotation_types, side_place_flag=self.side_place_flag)
                if len(action_list) == 0:
                    return 0.0
                else:
                    for action in action_list:
                        child_action_node = self.Tree.number_of_nodes()
                        self.Tree.add_node(child_action_node)
                        if self.with_robot:
                            self.Tree.update(nodes=[(child_action_node,
                                                     {'depth': depth,
                                                      'state': obj_list,
                                                      'left_joint_values': left_joint_values,
                                                      'left_gripper_width': left_gripper_width,
                                                      'right_joint_values': right_joint_values,
                                                      'right_gripper_width': right_gripper_width,
                                                      'action': action,
                                                      'reward': 0.,
                                                      'value': -np.inf,
                                                      'visit': 0})])
                        else:
                            self.Tree.update(nodes=[(child_action_node,
                                                     {'depth': depth,
                                                      'state': obj_list,
                                                      'action': action,
                                                      'reward': 0.,
                                                      'value': -np.inf,
                                                      'visit': 0})])
                        self.Tree.add_edge(state_node, child_action_node)
                    action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]

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

    def action_exploration(self, action_node):
        obj_list = self.Tree.nodes[action_node]['state']
        action = self.Tree.nodes[action_node]['action']
        depth = self.Tree.nodes[action_node]['depth']
        visit = self.Tree.nodes[action_node]['visit']
        if self.with_robot:
            left_joint_values = self.Tree.nodes[action_node]['left_joint_values']
            left_gripper_width = self.Tree.nodes[action_node]['left_gripper_width']
            right_joint_values = self.Tree.nodes[action_node]['right_joint_values']
            right_gripper_width = self.Tree.nodes[action_node]['right_gripper_width']
        self.Tree.update(nodes=[(action_node, {'visit': visit + 1})])

        next_state_nodes = [node for node in self.Tree.neighbors(action_node)]
        if len(next_state_nodes) == 0:
            if self.with_robot:
                next_obj_list, next_left_joint_values, next_left_gripper_width, next_right_joint_values, next_right_gripper_width, planned_traj_list = \
                    get_transition_with_robot(obj_list, left_joint_values, left_gripper_width,
                                              right_joint_values, right_gripper_width,
                                              action, self.meshes,
                                              _get_planning_scene_proxy=self.get_planning_scene_proxy,
                                              _apply_planning_scene_proxy=self.apply_planning_scene_proxy,
                                              _compute_fk_proxy=self.compute_fk_proxy,
                                              _planning_with_gripper_pose_proxy=self.planning_with_gripper_pose_proxy,
                                              _planning_with_arm_joints_proxy=self.planning_with_arm_joints_proxy)
            else:
                next_obj_list = get_transition(obj_list, action)

            if next_obj_list is not None:
                if self.physcial_constraint_checker is not None:
                    physical_demonstratablity = self.physcial_constraint_checker(obj_list, action, next_obj_list, self.meshes, network=self.network)
                else:
                    physical_demonstratablity = True

                if physical_demonstratablity:
                    rew = get_reward(obj_list, action, self.goal_obj, next_obj_list, self.meshes)

                    child_node = self.Tree.number_of_nodes()
                    self.Tree.add_node(child_node)
                    if self.with_robot:
                        self.Tree.update(nodes=[(child_node,
                                                 {'depth': depth + 1,
                                                  'state': next_obj_list,
                                                  'left_joint_values': next_left_joint_values,
                                                  'left_gripper_width': next_left_gripper_width,
                                                  'right_joint_values': next_right_joint_values,
                                                  'right_gripper_width': next_right_gripper_width,
                                                  'planned_traj_list': planned_traj_list,
                                                  'reward': rew,
                                                  'value': -np.inf,
                                                  'visit': 0})])
                    else:
                        self.Tree.update(nodes=[(child_node,
                                                 {'depth': depth + 1,
                                                  'state': next_obj_list,
                                                  'reward': rew,
                                                  'value': -np.inf,
                                                  'visit': 0})])
                    self.Tree.add_edge(action_node, child_node)
            next_state_nodes = [node for node in self.Tree.neighbors(action_node)]

        if len(next_state_nodes) > 0:
            next_state_node = next_state_nodes[0]
            next_state_value = self.Tree.nodes[next_state_node]['value']
            next_state_visit = self.Tree.nodes[next_state_node]['visit']
            reward = self.Tree.nodes[next_state_node]['reward']

            next_state_value_new = reward + self.exploration(next_state_node)
            if next_state_value < next_state_value_new:
                next_state_value = next_state_value_new
        else:
            next_state_value = -np.inf
            self.Tree.update(nodes=[(action_node, {'reward': -np.inf})])
        self.Tree.update(nodes=[(action_node, {'value': next_state_value})])
        return next_state_value

    def get_best_path(self, start_node=0):
        next_nodes = [next_node for next_node in self.Tree.neighbors(start_node)]
        if len(next_nodes) == 0:
            return [start_node]
        else:
            best_idx = np.argmax([self.Tree.nodes[next_node]['value'] for next_node in next_nodes])
            next_node = next_nodes[best_idx]
            return [start_node, ] + self.get_best_path(next_node)

    def visualize(self):
        depths = [self.Tree.nodes[n]['depth'] for n in self.Tree.nodes]
        visits = [self.Tree.nodes[n]['visit'] for n in self.Tree.nodes]
        rewards = [self.Tree.nodes[n]['reward'] for n in self.Tree.nodes]
        values = [self.Tree.nodes[n]['value'] for n in self.Tree.nodes]
        labels = {
            n: 'depth:{:d}\nvisit:{:d}\nreward:{:.4f}\nvalue:{:.4f}'.format(depths[n], visits[n], rewards[n], values[n])
            for n in self.Tree.nodes}

        nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        # same layout using matplotlib with no labels
        plt.figure(figsize=(16, 32))
        plt.title('')
        nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        pos = graphviz_layout(self.Tree, prog='dot')
        nx.draw(self.Tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        plt.show()


if __name__ == '__main__':
    with_robot = False
    if with_robot:
        initial_left_joint_values = {'left_w0': 0.6699952259595108,
                                     'left_w1': 1.030009435085784,
                                     'left_w2': -0.4999997247485215,
                                     'left_e0': -1.189968899785275,
                                     'left_e1': 1.9400238130755056,
                                     'left_s0': -0.08000397926829805,
                                     'left_s1': -0.9999781166910306}
        initial_left_gripper_width = 0.03
        initial_right_joint_values = {'right_w0': -0.6699952259595108,
                                      'right_w1': 1.030009435085784,
                                      'right_w2': 0.4999997247485215,
                                      'right_e0': 1.189968899785275,
                                      'right_e1': 1.9400238130755056,
                                      'right_s0': -0.08000397926829805,
                                      'right_s1': -0.9999781166910306}
        initial_right_gripper_width = 0.03

        rospy.init_node('mcts_moveit_planner_unit_test', anonymous=True)
        rospy.wait_for_service('/get_planning_scene')
        rospy.wait_for_service('/apply_planning_scene')
        rospy.wait_for_service('/compute_ik')
        rospy.wait_for_service('/compute_fk')

        get_planning_scene_proxy = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        apply_planning_scene_proxy = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
        compute_fk_proxy = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        planning_with_gripper_pose_proxy = rospy.ServiceProxy('/planning_with_gripper_pose', MoveitPlanningGripperPose)
        planning_with_arm_joints_proxy = rospy.ServiceProxy('/planning_with_arm_joints', MoveitPlanningJointValues)

        area_ths = 1.
    else:
        get_planning_scene_proxy = None
        apply_planning_scene_proxy = None
        compute_fk_proxy = None
        planning_with_gripper_pose_proxy = None
        planning_with_arm_joints_proxy = None

        area_ths = 0.003

    mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=area_ths)
    initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
        configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
                                  goal_name='stack_easy')

    n_seed = 10
    opt_num = 500

    seed_value_list = []
    seed_value_indices = []
    seed_final_state_list = []

    for seed in range(n_seed):
        mcts = Tree(initial_object_list, np.sum(n_obj_per_mesh_types) * 2, coll_mngr, meshes, contact_points,
                    contact_faces, rotation_types, _goal_obj=goal_obj,
                    _get_planning_scene_proxy=get_planning_scene_proxy,
                    _apply_planning_scene_proxy=apply_planning_scene_proxy,
                    _compute_fk_proxy=compute_fk_proxy,
                    _planning_with_gripper_pose_proxy=planning_with_gripper_pose_proxy,
                    _planning_with_arm_joints_proxy=planning_with_arm_joints_proxy,
                    _with_robot=with_robot)
        best_value_indices = []
        best_value_list = []
        best_final_state_list = []

        best_value = -np.inf
        print('START : {}th seed'.format(seed))
        for opt_idx in range(opt_num):
            mcts.exploration(0)
            if best_value < mcts.Tree.nodes[0]['value']:
                best_value = mcts.Tree.nodes[0]['value']
                best_value_list.append(best_value)
                best_value_indices.append(opt_idx)

                best_path_indices = mcts.get_best_path()
                best_final_state_list.append(mcts.Tree.nodes[best_path_indices[-1]]['state'])

                print(opt_idx, best_value)
            if (opt_idx + 1) % 100 == 0:
                print('============={}/{}============='.format(opt_idx, opt_num))

        seed_value_list.append(best_value_list)
        seed_value_indices.append(best_value_indices)
        seed_final_state_list.append(best_final_state_list)
        print('DONE : {}th seed'.format(seed))
