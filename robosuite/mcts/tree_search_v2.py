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


def check_stability(obj_idx, object_list, meshes, com1, volume1):
    if "on" in object_list[obj_idx].logical_state:
        obj_name = object_list[obj_idx].logical_state["on"][0]
        support_obj_idx = get_obj_idx_by_name(object_list, obj_name)
        mesh2 = deepcopy(meshes[object_list[support_obj_idx].mesh_idx])
        mesh2.apply_transform(object_list[support_obj_idx].pose)

        closest_points, dists, surface_idx = trimesh.proximity.closest_point(mesh2, [com1])
        project_point2 = closest_points[0]
        project_point2 = project_point2 / np.sqrt(np.sum(project_point2 ** 2.))
        safe_com = mesh2.face_normals[surface_idx[0]][2] > 0.999 and (project_point2[0] > mesh2.bounds[0][0] + 0.01) and (project_point2[0] < mesh2.bounds[1][0] - 0.01) and (project_point2[1] > mesh2.bounds[0][1] + 0.01) and (project_point2[1] < mesh2.bounds[1][1] - 0.01)
        # print(dists[0])
        # print(com1)
        # print(closest_points[0])
        # print(project_point2)
        # print(mesh2.bounds[0][0]+0.01, project_point2[0])
        # print(safe_com, obj_name)
        if safe_com:
        # if True:
            com1 = (com1*volume1 + mesh2.center_mass*mesh2.volume)/(volume1+mesh2.volume)
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
        if len(selected_visits) == 1:
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
        if len(selected_visits) == 1:
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
        self.KinematicTree = None

        self.max_depth = _max_depth
        self.min_visit = _min_visit
        self.Tree.add_node(0)
        self.Tree.update(nodes=[(0, {'depth': 0,
                                     'state': _init_obj_list,
                                     'left_joint_values': _init_left_joint_values,
                                     'left_gripper_width': _init_left_gripper_width,
                                     'right_joint_values': _init_right_joint_values,
                                     'right_gripper_width': _init_right_gripper_width,
                                     'reward': 0,
                                     'true_reward': 0.,
                                     'value': -np.inf,
                                     'true_value': -np.inf,
                                     'visit': 0,
                                     'true_visit': 0})])

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
        self.Tree.update(nodes=[(state_node, {'visit': visit + 1})])

        if depth < self.max_depth:
            obj_list = self.Tree.nodes[state_node]['state']
            action_nodes = [action_node for action_node in self.Tree.neighbors(state_node) if self.Tree.nodes[action_node]['reward'] == 0.]
            if obj_list is None:
                return 0.0
            elif len(action_nodes) == 0:
                action_list = get_possible_actions_v2(obj_list, self.meshes, self.coll_mngr,
                                                   self.contact_points, self.contact_faces, self.rotation_types, side_place_flag=self.side_place_flag)
                if len(action_list) == 0:
                    return 0.0
                else:
                    for action in action_list:
                        child_action_node = self.Tree.number_of_nodes()
                        self.Tree.add_node(child_action_node)
                        if 'place' in action['type']:
                            self.Tree.update(nodes=[(child_action_node,
                                                     {'depth': depth,
                                                      'state': obj_list,
                                                      'action': action,
                                                      'reward': 0.,
                                                      'true_reward': 0.,
                                                      'value': -np.inf,
                                                      'true_value': -np.inf,
                                                      'done': False,
                                                      'visit': 0,
                                                      'true_visit': 0})])
                        if 'pick' in action['type']:
                            self.Tree.update(nodes=[(child_action_node,
                                                     {'depth': depth,
                                                      'state': obj_list,
                                                      'action': action,
                                                      'reward': 0.,
                                                      'true_reward': 0.,
                                                      'value': -np.inf,
                                                      'true_value': -np.inf,
                                                      'done': [],
                                                      'visit': 0,
                                                      'true_visit': 0})])
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
                self.Tree.update(nodes=[(state_node, {'true_value': np.max(action_values)})])
            return np.max(action_values)
        else:
            return 0.0

    def action_exploration(self, action_node):
        obj_list = self.Tree.nodes[action_node]['state']
        action = self.Tree.nodes[action_node]['action']
        depth = self.Tree.nodes[action_node]['depth']
        visit = self.Tree.nodes[action_node]['visit']
        self.Tree.update(nodes=[(action_node, {'visit': visit + 1})])

        next_state_nodes = [node for node in self.Tree.neighbors(action_node)]
        if len(next_state_nodes) == 0:
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
                    self.Tree.update(nodes=[(child_node,
                                             {'depth': depth + 1,
                                              'state': next_obj_list,
                                              'reward': rew,
                                              'true_reward': rew,
                                              'value': -np.inf,
                                              'true_value': -np.inf,
                                              'visit': 0,
                                              'true_visit': 0})])
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
        self.Tree.update(nodes=[(action_node, {'true_value': next_state_value})])
        return next_state_value

    def update_subtree(self):
        _visited_nodes = [n for n in self.Tree.nodes if self.Tree.nodes[n]['depth'] == self.max_depth]
        if len(_visited_nodes) == 0:
            _max_depth = np.max([self.Tree.nodes[n]['depth'] for n in self.Tree.nodes])
            _visited_nodes = [n for n in self.Tree.nodes if self.Tree.nodes[n]['depth'] == _max_depth]

        _children_nodes = _visited_nodes
        while len(_children_nodes) > 0:
            _parent_nodes = [parent for child in _children_nodes for parent in self.Tree.predecessors(child)]
            _visited_nodes += _parent_nodes
            _children_nodes = _parent_nodes

        _visited_nodes = np.unique(_visited_nodes)
        self.KinematicTree = self.Tree.subgraph(_visited_nodes)
        self.num_kinematic_leaves = len(_visited_nodes)

    def kinematic_exploration(self, state_node=0):
        depth = self.KinematicTree.nodes[state_node]['depth']
        visit = self.KinematicTree.nodes[state_node]['visit']
        obj_list = self.KinematicTree.nodes[state_node]['state']
        left_joint_values = self.KinematicTree.nodes[state_node]['left_joint_values']
        left_gripper_width = self.KinematicTree.nodes[state_node]['left_gripper_width']
        right_joint_values = self.KinematicTree.nodes[state_node]['right_joint_values']
        right_gripper_width = self.KinematicTree.nodes[state_node]['right_gripper_width']
        action_nodes = [action_node for action_node in self.KinematicTree.neighbors(state_node) if
                        self.KinematicTree.nodes[action_node]['true_reward'] == 0.]
        print("action length : ", state_node, len([action_node for action_node in self.KinematicTree.neighbors(state_node)]))
        print("available action length : ",state_node, len(action_nodes))

        if len(action_nodes) > 0:
            action_values = [self.KinematicTree.nodes[action_node]['true_value'] for action_node in action_nodes]
            action_visits = [self.KinematicTree.nodes[action_node]['visit'] for action_node in action_nodes]
            action_list = [self.KinematicTree.nodes[action_node]['action'] for action_node in action_nodes]

            eps = np.maximum(np.minimum(1., 1 / np.maximum(visit, 1)), 0.01)
            if np.any(['place' in action['type'] for action in action_list]):
                if self.goal_obj is not None:
                    table_place_indices = [action_idx for action_idx, action in enumerate(action_list) if
                                           'table' in action['param']]
                    if len(table_place_indices) > 0:
                        selected_idx = sampler(self.exploration_method, action_values, action_visits, depth,
                                               _indices=table_place_indices, eps=eps)
                    else:
                        selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=eps)
                else:
                    non_table_place_indices = [action_idx for action_idx, action in enumerate(action_list) if
                                               'table' not in action['param']]
                    if len(non_table_place_indices) > 0:
                        selected_idx = sampler(self.exploration_method, action_values, action_visits, depth,
                                               _indices=non_table_place_indices, eps=eps)
                    else:
                        selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=eps)
            else:
                selected_idx = sampler(self.exploration_method, action_values, action_visits, depth, eps=eps)

            selected_action_node = action_nodes[selected_idx]
            selected_action = action_list[selected_idx]
            selected_action_value = action_values[selected_idx]

            next_state_nodes = [node for node in self.KinematicTree.neighbors(selected_action_node)]
            if len(next_state_nodes) > 0:
                next_state_node = next_state_nodes[0]
                next_object_list = self.KinematicTree.nodes[next_state_node]['state']

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
                else:
                    print("planning passed since it is done")
                    new_next_object_list = None

                if self.KinematicTree.nodes[selected_action_node]['done'] or new_next_object_list is not None:
                    if new_next_object_list is not None:
                        self.Tree.update(nodes=[(next_state_node, {'left_joint_values': next_left_joint_values})])
                        self.Tree.update(nodes=[(next_state_node, {'left_gripper_width': next_left_gripper_width})])
                        self.Tree.update(nodes=[(next_state_node, {'right_joint_values': next_right_joint_values})])
                        self.Tree.update(nodes=[(next_state_node, {'right_gripper_width': next_right_gripper_width})])
                        self.Tree.update(nodes=[(next_state_node, {'planned_traj_list': planned_traj_list})])

                    if selected_action['type'] is 'pick':
                        self.Tree.update(nodes=[(next_state_node, {'place_fail': False})])

                    if len([n for n in self.Tree.neighbors(next_state_node)]) > 0:
                        true_action_value = self.KinematicTree.nodes[next_state_node]['true_reward'] + self.kinematic_exploration(next_state_node)
                    else:
                        print("finish here4", state_node, selected_action_node)
                        self.Tree.update(nodes=[(selected_action_node, {'done': True})])
                        true_action_value = self.KinematicTree.nodes[next_state_node]['true_reward']

                    if selected_action_value < true_action_value:
                        selected_action_value = true_action_value
                        self.Tree.update(nodes=[(selected_action_node, {'true_value': selected_action_value})])
                        self.Tree.update(nodes=[(selected_action_node, {'done': True})])
                    elif np.isinf(true_action_value) and selected_action['type'] is 'pick' and self.Tree.nodes[next_state_node]['place_fail']:
                        print("successfully grasp, but, placing failed", selected_action["search_idx"])
                        self.Tree.update(nodes=[(next_state_node, {'place_fail': False})])
                        selected_action["search_idx"] += 1
                        if selected_action["search_idx"] < len(selected_action["grasp_poses"]):
                            self.Tree.update(nodes=[(selected_action_node, {'action': selected_action})])
                        else:
                            self.Tree.update(nodes=[(selected_action_node, {'true_reward': -np.inf})])
                            self.Tree.update(nodes=[(selected_action_node, {'true_value': -np.inf})])
                    elif np.isinf(true_action_value) and selected_action['type'] is 'pick' and not self.Tree.nodes[next_state_node]['place_fail']:
                        if new_next_object_list is not None:
                            self.Tree.update(nodes=[(next_state_node, {'state': new_next_object_list})])
                        self.Tree.update(nodes=[(selected_action_node, {'done': True})])
                    elif np.isinf(true_action_value) and selected_action['type'] is 'place' and new_next_object_list is not None:
                        self.Tree.update(nodes=[(selected_action_node, {'done': True})])

                    return selected_action_value
                else:
                    if selected_action['type'] is 'pick':
                        print("grasp failed", state_node, selected_action_node, selected_action["search_idx"])
                        selected_action["search_idx"] += 1
                        if selected_action["search_idx"] < len(selected_action["grasp_poses"]):
                            self.Tree.update(nodes=[(selected_action_node, {'action': selected_action})])
                        else:
                            print("finish here4", state_node, selected_action_node)
                            self.Tree.update(nodes=[(selected_action_node, {'true_reward': -np.inf})])
                            self.Tree.update(nodes=[(selected_action_node, {'true_value': -np.inf})])
                    else:
                        print("finish here3")
                        self.Tree.update(nodes=[(state_node, {'place_fail': True})])
                    return -np.inf
            else:
                print("finish here2", state_node, selected_action_node)
                if selected_action['type'] is 'place':
                    self.Tree.update(nodes=[(state_node, {'place_fail': True})])
                self.Tree.update(nodes=[(selected_action_node, {'true_reward': -np.inf})])
                self.Tree.update(nodes=[(selected_action_node, {'true_value': -np.inf})])
                return -np.inf
        else:
            print("finish here1 at state ", state_node)
            self.Tree.update(nodes=[(state_node, {'true_reward': -np.inf})])
            self.Tree.update(nodes=[(state_node, {'true_value': -np.inf})])
            return -np.inf

    def kinematic_exploration_v2(self, state_node=0):
        depth = self.KinematicTree.nodes[state_node]['depth']
        visit = self.KinematicTree.nodes[state_node]['true_visit']
        state_value = self.KinematicTree.nodes[state_node]['true_value']
        obj_list = self.KinematicTree.nodes[state_node]['state']
        left_joint_values = self.KinematicTree.nodes[state_node]['left_joint_values']
        left_gripper_width = self.KinematicTree.nodes[state_node]['left_gripper_width']
        right_joint_values = self.KinematicTree.nodes[state_node]['right_joint_values']
        right_gripper_width = self.KinematicTree.nodes[state_node]['right_gripper_width']
        self.Tree.update(nodes=[(state_node, {'true_visit': visit + 1})])

        action_nodes = [action_node for action_node in self.KinematicTree.neighbors(state_node) if self.KinematicTree.nodes[action_node]['true_reward'] == 0.]
        action_values = [self.KinematicTree.nodes[action_node]['true_value'] for action_node in action_nodes]
        action_visits = [self.KinematicTree.nodes[action_node]['true_visit'] for action_node in action_nodes]
        action_list = [self.KinematicTree.nodes[action_node]['action'] for action_node in action_nodes]
        print("============= pick node information ============")
        print(depth, "/", self.max_depth, "state node :", state_node)
        print(depth, "/", self.max_depth, "possible actions / total actions :", len(action_nodes), len([action_node for action_node in self.KinematicTree.neighbors(state_node)]))
        if len(action_nodes) > 0:
            selected_idx = sampler({'method': 'greedy'}, action_values, action_visits, depth, eps=0.)
            selected_action_node = action_nodes[selected_idx]
            selected_action_value = action_values[selected_idx]
            selected_action_visit = action_visits[selected_idx]
            selected_action = action_list[selected_idx]
            self.Tree.update(nodes=[(selected_action_node, {'true_visit': selected_action_visit + 1})])

            next_state_nodes = [node for node in self.KinematicTree.neighbors(selected_action_node)]
            next_state_node = next_state_nodes[0]
            next_depth = self.KinematicTree.nodes[next_state_node]['depth']
            next_visit = self.KinematicTree.nodes[next_state_node]['true_visit']
            next_state_value = self.KinematicTree.nodes[next_state_node]['true_value']
            next_object_list = self.KinematicTree.nodes[next_state_node]['state']
            self.Tree.update(nodes=[(next_state_node, {'true_visit': next_visit + 1})])

            next_action_nodes = [next_action_node for next_action_node in self.KinematicTree.neighbors(next_state_node) if self.KinematicTree.nodes[next_action_node]['true_reward'] == 0.]
            next_action_values = [self.KinematicTree.nodes[next_action_node]['true_value'] for next_action_node in next_action_nodes]
            next_action_visits = [self.KinematicTree.nodes[next_action_node]['true_visit'] for next_action_node in next_action_nodes]
            next_action_list = [self.KinematicTree.nodes[next_action_node]['action'] for next_action_node in next_action_nodes]

            print("============= place node information ============")
            print(next_depth, "/", self.max_depth, "state node :", next_state_node)
            print(next_depth, "/", self.max_depth, "possible actions / total actions :", len(next_action_nodes), len([next_action_node for next_action_node in self.KinematicTree.neighbors(next_state_node)]))
            if len(next_action_nodes) > 0:
                next_selected_idx = sampler({'method': 'greedy'}, next_action_values, next_action_visits, next_depth, eps=0.)
                selected_next_action_node = next_action_nodes[next_selected_idx]
                selected_next_action_value = next_action_values[next_selected_idx]
                selected_next_action_visit = next_action_visits[selected_idx]
                selected_next_action = next_action_list[next_selected_idx]
                self.Tree.update(nodes=[(selected_next_action_node, {'true_visit': selected_next_action_visit + 1})])

                next_next_state_nodes = [node for node in self.KinematicTree.neighbors(selected_next_action_node)]
                next_next_state_node = next_next_state_nodes[0]
                next_next_object_list = self.KinematicTree.nodes[next_next_state_node]['state']

                search_idx = 0
                done = False
                print("pick planning is done before: ", self.KinematicTree.nodes[selected_action_node]['done'])
                print("place planning is done before: ", self.KinematicTree.nodes[selected_next_action_node]['done'])

                # while search_idx < len(selected_action["grasp_poses"]):
                print("search idx : ", search_idx)
                pick_planning = False
                place_list = self.KinematicTree.nodes[selected_action_node]['done']
                if selected_next_action_node not in place_list:
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
                        pick_planning = True
                else:
                    pick_planning = True

                if pick_planning:
                    place_planning = False
                    if not self.KinematicTree.nodes[selected_next_action_node]['done']:
                        new_next_next_object_list, next_next_left_joint_values, next_next_left_gripper_width, next_next_right_joint_values, next_next_right_gripper_width, next_planned_traj_list = \
                            kinematic_planning(next_object_list, next_next_object_list,
                                               next_left_joint_values, next_left_gripper_width,
                                               next_right_joint_values, next_right_gripper_width,
                                               selected_next_action, self.meshes,
                                               _get_planning_scene_proxy=self.get_planning_scene_proxy,
                                               _apply_planning_scene_proxy=self.apply_planning_scene_proxy,
                                               _cartesian_planning_with_gripper_pose_proxy=self.cartesian_planning_with_gripper_pose_proxy,
                                               _planning_with_gripper_pose_proxy=self.planning_with_gripper_pose_proxy,
                                               _planning_with_arm_joints_proxy=self.planning_with_arm_joints_proxy,
                                               _compute_fk_proxy=self.compute_fk_proxy)
                        if new_next_next_object_list is not None:
                            place_planning = True
                    else:
                        place_planning = True
                if pick_planning and place_planning:
                    done = True
                    #     break
                    # else:
                    #     search_idx += 1
                if done:
                    if selected_next_action_node not in self.KinematicTree.nodes[selected_action_node]['done']:
                        if 'planned_traj_list' not in self.Tree.nodes[next_state_node]:
                            self.Tree.update(nodes=[(next_state_node, {'left_joint_values': [next_left_joint_values]})])
                            self.Tree.update(nodes=[(next_state_node, {'left_gripper_width': [next_left_gripper_width]})])
                            self.Tree.update(nodes=[(next_state_node, {'right_joint_values': [next_right_joint_values]})])
                            self.Tree.update(nodes=[(next_state_node, {'right_gripper_width': [next_right_gripper_width]})])
                            self.Tree.update(nodes=[(next_state_node, {'planned_traj_list': [planned_traj_list]})])
                        else:
                            left_joint_values_list = self.Tree.nodes[next_state_node]['left_joint_values']
                            left_gripper_width_list = self.Tree.nodes[next_state_node]['left_gripper_width']
                            right_joint_values_list = self.Tree.nodes[next_state_node]['right_joint_values']
                            right_gripper_width_list = self.Tree.nodes[next_state_node]['right_gripper_width']
                            planned_traj_list_list = self.Tree.nodes[next_state_node]['planned_traj_list']
                            left_joint_values_list.append(left_joint_values)
                            left_gripper_width_list.append(left_gripper_width)
                            right_joint_values_list.append(right_joint_values)
                            right_gripper_width_list.append(right_gripper_width)
                            planned_traj_list_list.append(planned_traj_list)
                            self.Tree.update(nodes=[(next_state_node, {'left_joint_values': left_joint_values_list})])
                            self.Tree.update(nodes=[(next_state_node, {'left_gripper_width': left_gripper_width_list})])
                            self.Tree.update(nodes=[(next_state_node, {'right_joint_values': right_joint_values_list})])
                            self.Tree.update(nodes=[(next_state_node, {'right_gripper_width': right_gripper_width_list})])
                            self.Tree.update(nodes=[(next_state_node, {'planned_traj_list': planned_traj_list_list})])

                        place_list.append(selected_next_action_node)
                        self.Tree.update(nodes=[(selected_action_node, {'done': place_list})])

                    if not self.KinematicTree.nodes[selected_next_action_node]['done']:
                        self.Tree.update(nodes=[(next_next_state_node, {'left_joint_values': next_next_left_joint_values})])
                        self.Tree.update(nodes=[(next_next_state_node, {'left_gripper_width': next_next_left_gripper_width})])
                        self.Tree.update(nodes=[(next_next_state_node, {'right_joint_values': next_next_right_joint_values})])
                        self.Tree.update(nodes=[(next_next_state_node, {'right_gripper_width': next_next_right_gripper_width})])
                        self.Tree.update(nodes=[(next_next_state_node, {'planned_traj_list': next_planned_traj_list})])
                        self.Tree.update(nodes=[(selected_next_action_node, {'done': True})])

                    new_next_action_value = self.KinematicTree.nodes[next_next_state_node]['true_reward'] + self.kinematic_exploration_v2(next_next_state_node)
                    # if len([next_next_action_node for next_next_action_node in self.KinematicTree.neighbors(next_next_state_node) if self.KinematicTree.nodes[next_next_action_node]['true_reward'] == 0.]) == 0:
                    #     self.Tree.update(nodes=[(selected_next_action_node, {'true_reward': -np.inf})])
                    #     self.Tree.update(nodes=[(selected_next_action_node, {'true_value': -np.inf})])

                    if selected_next_action_value < new_next_action_value:
                        selected_next_action_value = new_next_action_value
                        self.Tree.update(nodes=[(selected_next_action_node, {'true_value': selected_next_action_value})])

                    if next_state_value < new_next_action_value:
                        next_state_value = new_next_action_value
                        self.Tree.update(nodes=[(next_state_node, {'true_value': next_state_value})])

                    new_action_value = self.KinematicTree.nodes[next_state_node]['true_reward'] + new_next_action_value
                    if selected_action_value < new_action_value:
                        selected_action_value = new_action_value
                        self.Tree.update(nodes=[(selected_action_node, {'true_value': selected_action_value})])

                    if state_value < new_next_action_value:
                        state_value = new_next_action_value
                        self.Tree.update(nodes=[(state_node, {'true_value': state_value})])
                    return state_value
                else:
                    self.Tree.update(nodes=[(selected_next_action_node, {'true_reward': -np.inf})])
                    self.Tree.update(nodes=[(selected_next_action_node, {'true_value': -np.inf})])
                    self.Tree.update(nodes=[(next_next_state_node, {'true_reward': -np.inf})])
                    self.Tree.update(nodes=[(next_next_state_node, {'true_value': -np.inf})])
                    return -np.inf
            else:
                self.Tree.update(nodes=[(next_state_node, {'true_reward': -np.inf})])
                self.Tree.update(nodes=[(next_state_node, {'true_value': -np.inf})])
                self.Tree.update(nodes=[(selected_action_node, {'true_reward': -np.inf})])
                self.Tree.update(nodes=[(selected_action_node, {'true_value': -np.inf})])
                return -np.inf
        else:
            print("There is no possible pick action")
            return 0.

    def get_best_path(self, start_node=0):
        next_nodes = [next_node for next_node in self.Tree.neighbors(start_node)]
        if len(next_nodes) == 0:
            return [start_node]
        else:
            best_idx = np.argmax([self.Tree.nodes[next_node]['value'] for next_node in next_nodes])
            next_node = next_nodes[best_idx]
            return [start_node, ] + self.get_best_path(next_node)

    def get_best_kinematic_path(self, start_node=0):
        next_nodes = [next_node for next_node in self.KinematicTree.neighbors(start_node) if not np.isinf(self.KinematicTree.nodes[next_node]['true_reward'])]
        if len(next_nodes) == 0:
            return [start_node]
        else:
            best_idx = np.argmax([self.KinematicTree.nodes[next_node]['true_value'] for next_node in next_nodes])
            next_node = next_nodes[best_idx]
            return [start_node, ] + self.get_best_kinematic_path(next_node)

    def visualize(self):
        visited_nodes = [n for n in self.Tree.nodes if self.Tree.nodes[n]['visit'] > 0]
        visited_tree = self.Tree.subgraph(visited_nodes)
        # depths = [visited_tree.nodes[n]['depth'] for n in visited_tree.nodes]
        # visits = [visited_tree.nodes[n]['visit'] for n in visited_tree.nodes]
        # rewards = [visited_tree.nodes[n]['reward'] for n in visited_tree.nodes]
        # values = [visited_tree.nodes[n]['value'] for n in visited_tree.nodes]
        labels = {
            n: 'depth:{:d}\nvisit:{:d}\nreward:{:.4f}\nvalue:{:.4f}'.format(visited_tree.nodes[n]['depth'], visited_tree.nodes[n]['visit'], visited_tree.nodes[n]['reward'], visited_tree.nodes[n]['value'])
            for n in visited_tree.nodes}

        # nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        # same layout using matplotlib with no labels
        # plt.figure(figsize=(16, 32))
        plt.figure()
        plt.title('')
        # nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        pos = graphviz_layout(visited_tree, prog='dot')
        nx.draw(visited_tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        plt.show()

    def visualize_tree(self, visited_tree):
        labels = {
            n: 'depth:{:d}\nvisit:{:d}\nreward:{:.4f}\nvalue:{:.4f}'.format(visited_tree.nodes[n]['depth'], visited_tree.nodes[n]['visit'], visited_tree.nodes[n]['reward'], visited_tree.nodes[n]['value'])
            for n in visited_tree.nodes}

        plt.figure()
        plt.title('')

        pos = graphviz_layout(visited_tree, prog='dot')
        nx.draw(visited_tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        plt.show()


if __name__ == '__main__':
    mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=1.)
    initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
        configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
                                  goal_name='debug_config')

    # n_seed = 1
    # opt_num = 100
    #
    # seed_value_list = []
    # seed_value_indices = []
    # seed_final_state_list = []
    #
    # for seed in range(n_seed):
    #     mcts = Tree(initial_object_list, np.sum(n_obj_per_mesh_types) * 2, coll_mngr, meshes, contact_points,
    #                 contact_faces, rotation_types, _goal_obj=goal_obj)
    #     best_value_indices = []
    #     best_value_list = []
    #     best_final_state_list = []
    #
    #     best_value = -np.inf
    #     print('START : {}th seed'.format(seed))
    #     for opt_idx in range(opt_num):
    #         mcts.exploration(0)
    #         if best_value < mcts.Tree.nodes[0]['value']:
    #             best_value = mcts.Tree.nodes[0]['value']
    #             best_value_list.append(best_value)
    #             best_value_indices.append(opt_idx)
    #
    #             best_path_indices = mcts.get_best_path()
    #             best_final_state_list.append(mcts.Tree.nodes[best_path_indices[-1]]['state'])
    #
    #             print(opt_idx, best_value)
    #         if (opt_idx + 1) % 100 == 0:
    #             print('============={}/{}============='.format(opt_idx, opt_num))
    #
    #     visualize(best_final_state_list[-1], meshes, _goal_obj=goal_obj)
    #
    #     seed_value_list.append(best_value_list)
    #     seed_value_indices.append(best_value_indices)
    #     seed_final_state_list.append(best_final_state_list)
    #     print('DONE : {}th seed'.format(seed))

    rospy.init_node('mcts_moveit_planner_unit_test', anonymous=True)
    rospy.wait_for_service('/get_planning_scene')
    rospy.wait_for_service('/apply_planning_scene')
    rospy.wait_for_service('/compute_fk')
    rospy.wait_for_service('/cartesian_planning_with_gripper_pose')
    rospy.wait_for_service('/planning_with_gripper_pose')
    rospy.wait_for_service('/planning_with_arm_joints')

    get_planning_scene_proxy = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
    apply_planning_scene_proxy = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
    compute_fk_proxy = rospy.ServiceProxy('/compute_fk', GetPositionFK)
    cartesian_planning_with_gripper_pose_proxy = rospy.ServiceProxy('/cartesian_planning_with_gripper_pose', MoveitPlanningGripperPose)
    planning_with_gripper_pose_proxy = rospy.ServiceProxy('/planning_with_gripper_pose', MoveitPlanningGripperPose)
    planning_with_arm_joints_proxy = rospy.ServiceProxy('/planning_with_arm_joints', MoveitPlanningJointValues)

    mcts = Tree(initial_object_list, np.sum(n_obj_per_mesh_types) * 2, coll_mngr, meshes, contact_points,
                contact_faces, rotation_types, _goal_obj=goal_obj,
                _get_planning_scene_proxy=get_planning_scene_proxy,
                _apply_planning_scene_proxy=apply_planning_scene_proxy,
                _compute_fk_proxy=compute_fk_proxy,
                _cartesian_planning_with_gripper_pose_proxy=cartesian_planning_with_gripper_pose_proxy,
                _planning_with_gripper_pose_proxy=planning_with_gripper_pose_proxy,
                _planning_with_arm_joints_proxy=planning_with_arm_joints_proxy)
    for opt_idx in range(10):
        mcts.exploration(0)
        print(opt_idx)

    best_path_indices = mcts.get_best_path(0)
    best_object_list = mcts.Tree.nodes[best_path_indices[-1]]['state']
    print(best_path_indices)

    mcts.update_subtree()
    for opt_idx in range(10):
        print(opt_idx, mcts.kinematic_exploration_v2(0))

    kinematic_best_path_indices = mcts.get_best_kinematic_path(0)
    print(kinematic_best_path_indices)



