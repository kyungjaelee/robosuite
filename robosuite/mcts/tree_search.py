import os
import sys
import copy
import random

import numpy as np

import rospy
from moveit_msgs.srv import *
from moveit_msgs.msg import *
from shape_msgs.msg import *
from geometry_msgs.msg import *
from mujoco_moveit_connector.srv import *

from robosuite.mcts.util import *
from robosuite.utils import transform_utils as tf
import trimesh.proximity

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import torch

from matplotlib import pyplot as plt


def transform_matrix2pose(T):
    q = tf.mat2quat(T)
    orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    position = Point(x=T[0, 3], y=T[1, 3], z=T[2, 3] - 0.93)
    pose = Pose(position, orientation)

    return pose


def pose2transform_matrix(pose):
    orientation = pose.orientation
    position = pose.position

    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    T = np.eye(4)
    T[:3, :3] = tf.quat2mat(q)
    T[0, 3] = position.x
    T[1, 3] = position.y
    T[2, 3] = position.z

    return T


def synchronize_planning_scene(left_joint_values,
                               right_joint_values,
                               object_list,
                               meshes,
                               left_gripper_width=0.0,
                               right_gripper_width=0.0,
                               get_planning_scene_proxy=None,
                               apply_planning_scene_proxy=None):
    joint_values = {}
    joint_values.update(left_joint_values)
    joint_values.update(right_joint_values)

    resp = get_planning_scene_proxy(GetPlanningSceneRequest())
    current_scene = resp.scene

    next_scene = deepcopy(current_scene)
    next_scene.robot_state.joint_state.name = list(joint_values.keys())
    next_scene.robot_state.joint_state.position = list(joint_values.values())
    next_scene.robot_state.joint_state.velocity = [0] * len(joint_values)
    next_scene.robot_state.joint_state.effort = [0] * len(joint_values)
    next_scene.robot_state.is_diff = True

    held_obj_idx = get_held_object(object_list)
    if held_obj_idx is None and len(next_scene.robot_state.attached_collision_objects) > 0:
        for attached_object_idx in range(len(next_scene.robot_state.attached_collision_objects)):
            next_scene.robot_state.attached_collision_objects[
                attached_object_idx].object.operation = CollisionObject.REMOVE
    if len(next_scene.world.collision_objects) > 0:
        for scene_object_idx in range(len(next_scene.world.collision_objects)):
            next_scene.world.collision_objects[scene_object_idx].operation = CollisionObject.REMOVE

    for obj in object_list:
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = obj.name
        co.header = next_scene.robot_state.joint_state.header

        mesh = Mesh()
        for face in meshes[obj.mesh_idx].faces:
            triangle = MeshTriangle()
            triangle.vertex_indices = face
            mesh.triangles.append(triangle)

        for vertex in meshes[obj.mesh_idx].vertices:
            point = Point()
            point.x = vertex[0]
            point.y = vertex[1]
            point.z = vertex[2]
            mesh.vertices.append(point)

        co.meshes = [mesh]

        obj_pose = transform_matrix2pose(obj.pose)
        co.mesh_poses = [obj_pose]

        if held_obj_idx is not None and len(next_scene.robot_state.attached_collision_objects) == 0 \
                and object_list[held_obj_idx].name is obj.name:
            aco = AttachedCollisionObject()
            aco.link_name = 'left_gripper'
            aco.object = co
            next_scene.robot_state.attached_collision_objects.append(aco)
        else:
            next_scene.world.collision_objects.append(co)

    next_scene.robot_state.joint_state.name += ('l_gripper_l_finger_joint', 'l_gripper_r_finger_joint')
    next_scene.robot_state.joint_state.position += (np.minimum(0.02, left_gripper_width), -np.minimum(0.02, left_gripper_width))

    next_scene.robot_state.joint_state.name += ('r_gripper_l_finger_joint', 'r_gripper_r_finger_joint')
    next_scene.robot_state.joint_state.position += (
    np.minimum(0.02, right_gripper_width), -np.minimum(0.02, right_gripper_width))

    next_scene.is_diff = True
    req = ApplyPlanningSceneRequest()
    req.scene = next_scene
    resp = apply_planning_scene_proxy(req)
    for _ in range(100):
        rospy.sleep(0.001)


def get_transition_with_baxter(left_arm_joint_values,
                               right_arm_joint_values,
                               left_arm_init_joint_values,
                               right_arm_init_joint_values,
                               object_list,
                               left_gripper_width,
                               right_gripper_width,
                               action,
                               meshes,
                               coll_mngr,
                               goal_obj=None,
                               get_planning_scene_proxy=None,
                               apply_planning_scene_proxy=None,
                               compute_fk_proxy=None,
                               mujoco_moveit_planner_proxy=None,
                               n_planning_trial=10):
    for obj in object_list:
        coll_mngr.set_transform(obj.name, obj.pose)

    synchronize_planning_scene(left_arm_joint_values,
                               right_arm_joint_values,
                               object_list,
                               meshes,
                               left_gripper_width=left_gripper_width,
                               right_gripper_width=right_gripper_width,
                               get_planning_scene_proxy=get_planning_scene_proxy,
                               apply_planning_scene_proxy=apply_planning_scene_proxy)

    planning_results = []
    planned_trajs = []
    planned_gripper_widths = []

    next_object_list = deepcopy(object_list)
    next_left_arm_joint_values = left_arm_joint_values
    next_right_arm_joint_values = right_arm_joint_values
    next_left_gripper_width = left_gripper_width
    next_right_gripper_width = right_gripper_width
    if action["type"] is "pick":
        pick_obj_idx = get_obj_idx_by_name(next_object_list, action['param'])
        for _ in range(n_planning_trial):
            next_gripper_pose, next_retreat_pose, _, next_gripper_width = sample_grasp_pose(
                next_object_list[pick_obj_idx], meshes)
            if next_gripper_pose is None: continue

            synchronize_planning_scene(left_arm_joint_values,
                                       right_arm_joint_values,
                                       next_object_list,
                                       meshes,
                                       left_gripper_width=next_gripper_width,
                                       right_gripper_width=right_gripper_width,
                                       get_planning_scene_proxy=get_planning_scene_proxy,
                                       apply_planning_scene_proxy=apply_planning_scene_proxy)
            moveit_req = MuJoCoMoveitConnectorRequest()
            moveit_req.planning_type = "gripper_pose"
            moveit_req.group_name = "left_arm"
            moveit_req.target_gripper_pose = transform_matrix2pose(next_retreat_pose)
            resp = mujoco_moveit_planner_proxy(moveit_req)
            approach_planning_result, approach_planned_traj = resp.success, resp.joint_trajectory
            if approach_planning_result:
                approach_left_arm_joint_values = {}
                for key, val in zip(approach_planned_traj.joint_names,
                                    approach_planned_traj.points[-1].positions):
                    approach_left_arm_joint_values[key] = val
                synchronize_planning_scene(approach_left_arm_joint_values,
                                           next_right_arm_joint_values,
                                           next_object_list,
                                           meshes,
                                           left_gripper_width=next_left_gripper_width,
                                           right_gripper_width=next_right_gripper_width,
                                           get_planning_scene_proxy=get_planning_scene_proxy,
                                           apply_planning_scene_proxy=apply_planning_scene_proxy)

                moveit_req = MuJoCoMoveitConnectorRequest()
                moveit_req.planning_type = "cartesian"
                moveit_req.group_name = "left_arm"
                moveit_req.target_gripper_pose = transform_matrix2pose(next_gripper_pose)
                resp = mujoco_moveit_planner_proxy(moveit_req)
                planning_result, planned_traj = resp.success, resp.joint_trajectory
                if planning_result:
                    next_left_arm_joint_values = {}
                    for key, val in zip(planned_traj.joint_names,
                                        planned_traj.points[-1].positions):
                        next_left_arm_joint_values[key] = val
                    next_left_gripper_width = next_gripper_width

                    next_right_arm_joint_values = right_arm_joint_values
                    next_right_gripper_width = right_gripper_width

                    next_object_list[pick_obj_idx].logical_state.clear()
                    next_object_list[pick_obj_idx].logical_state["held"] = []
                    update_logical_state(next_object_list)

                    synchronize_planning_scene(next_left_arm_joint_values,
                                               next_right_arm_joint_values,
                                               next_object_list,
                                               meshes,
                                               left_gripper_width=next_left_gripper_width,
                                               right_gripper_width=next_right_gripper_width,
                                               get_planning_scene_proxy=get_planning_scene_proxy,
                                               apply_planning_scene_proxy=apply_planning_scene_proxy)

                    planning_results.append(approach_planning_result)
                    planned_trajs.append(approach_planned_traj)

                    planning_results.append(planning_result)
                    planned_trajs.append(planned_traj)
                    planned_gripper_widths.append(next_left_gripper_width)
                    return next_object_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths

    if action["type"] is "place":
        moveit_req = MuJoCoMoveitConnectorRequest()
        moveit_req.planning_type = "joint_values"
        moveit_req.group_name = "left_arm"
        moveit_req.target_joint_names = left_arm_init_joint_values.keys()
        moveit_req.target_joint_values = left_arm_init_joint_values.values()
        resp = mujoco_moveit_planner_proxy(moveit_req)
        pre_planning_result, pre_planned_traj = resp.success, resp.joint_trajectory
        if pre_planning_result:
            synchronize_planning_scene(left_arm_init_joint_values,
                                       right_arm_joint_values,
                                       next_object_list,
                                       meshes,
                                       left_gripper_width=next_left_gripper_width,
                                       right_gripper_width=right_gripper_width,
                                       get_planning_scene_proxy=get_planning_scene_proxy,
                                       apply_planning_scene_proxy=apply_planning_scene_proxy)

            planning_results.append(pre_planning_result)
            planned_trajs.append(pre_planned_traj)
            planned_gripper_widths.append(left_gripper_width)

            held_obj_idx = get_held_object(next_object_list)
            resp = get_planning_scene_proxy(GetPlanningSceneRequest())
            current_scene = resp.scene
            rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
            rel_T = pose2transform_matrix(rel_obj_pose)

            req = GetPositionFKRequest()
            req.fk_link_names = ['left_gripper']
            req.header.frame_id = 'world'
            req.robot_state = current_scene.robot_state
            resp = compute_fk_proxy(req)

            gripper_T = pose2transform_matrix(resp.pose_stamped[0].pose)
            next_object_pose = gripper_T.dot(rel_T)
            next_object_pose[2, 3] += 0.93

            next_object_list[held_obj_idx].pose = next_object_pose

            place_obj_idx = get_obj_idx_by_name(next_object_list, action['param'])
            if "on" in next_object_list[held_obj_idx].logical_state:
                support_obj_idx = get_obj_idx_by_name(next_object_list,
                                                      next_object_list[held_obj_idx].logical_state["on"][0])
                next_object_list[support_obj_idx].logical_state["support"].remove(next_object_list[held_obj_idx].name)
                next_object_list[held_obj_idx].logical_state.pop("on")

            for _ in range(n_planning_trial):
                next_object_place_pose = sample_on_pose(next_object_list[place_obj_idx], next_object_list[held_obj_idx],
                                                        meshes, coll_mngr)
                if next_object_place_pose is None:
                    continue

                resp = get_planning_scene_proxy(GetPlanningSceneRequest())
                current_scene = resp.scene
                rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
                rel_T = pose2transform_matrix(rel_obj_pose)

                next_gripper_pose = next_object_place_pose.dot(np.linalg.inv(rel_T))
                moveit_req = MuJoCoMoveitConnectorRequest()
                moveit_req.planning_type = "gripper_pose"
                moveit_req.group_name = "left_arm"
                moveit_req.target_gripper_pose = transform_matrix2pose(next_gripper_pose)
                resp = mujoco_moveit_planner_proxy(moveit_req)
                planning_result, planned_traj = resp.success, resp.joint_trajectory
                if planning_result:
                    next_left_arm_joint_values = {}
                    for key, val in zip(planned_traj.joint_names,
                                        planned_traj.points[-1].positions):
                        next_left_arm_joint_values[key] = val
                    next_left_gripper_width = left_gripper_width

                    next_right_arm_joint_values = right_arm_joint_values
                    next_right_gripper_width = right_gripper_width

                    next_object_list[held_obj_idx].pose = next_object_place_pose
                    next_object_list[held_obj_idx].logical_state.clear()
                    next_object_list[held_obj_idx].logical_state["on"] = [next_object_list[place_obj_idx].name]
                    update_logical_state(next_object_list)

                    synchronize_planning_scene(next_left_arm_joint_values,
                                               next_right_arm_joint_values,
                                               next_object_list,
                                               meshes,
                                               left_gripper_width=next_left_gripper_width,
                                               right_gripper_width=next_right_gripper_width,
                                               get_planning_scene_proxy=get_planning_scene_proxy,
                                               apply_planning_scene_proxy=apply_planning_scene_proxy)
                    planning_results.append(planning_result)
                    planned_trajs.append(planned_traj)
                    planned_gripper_widths.append(next_left_gripper_width)

                    moveit_req = MuJoCoMoveitConnectorRequest()
                    moveit_req.planning_type = "joint_values"
                    moveit_req.group_name = "left_arm"
                    moveit_req.target_joint_names = left_arm_init_joint_values.keys()
                    moveit_req.target_joint_values = left_arm_init_joint_values.values()
                    resp = mujoco_moveit_planner_proxy(moveit_req)
                    after_planning_result, after_planned_traj = resp.success, resp.joint_trajectory
                    if after_planning_result:
                        next_left_arm_joint_values = {}
                        for key, val in zip(after_planned_traj.joint_names,
                                            after_planned_traj.points[-1].positions):
                            next_left_arm_joint_values[key] = val
                        next_left_gripper_width = left_gripper_width

                        next_right_arm_joint_values = right_arm_joint_values
                        next_right_gripper_width = right_gripper_width

                        synchronize_planning_scene(next_left_arm_joint_values,
                                                   next_right_arm_joint_values,
                                                   next_object_list,
                                                   meshes,
                                                   left_gripper_width=next_left_gripper_width,
                                                   right_gripper_width=next_right_gripper_width,
                                                   get_planning_scene_proxy=get_planning_scene_proxy,
                                                   apply_planning_scene_proxy=apply_planning_scene_proxy)
                        planning_results.append(after_planning_result)
                        planned_trajs.append(after_planned_traj)
                        planned_gripper_widths.append(next_left_gripper_width)
                        return next_object_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths

    planning_results = []
    planned_trajs = []
    planned_gripper_widths = []

    next_object_list = deepcopy(object_list)
    next_left_arm_joint_values = left_arm_joint_values
    next_right_arm_joint_values = right_arm_joint_values
    next_left_gripper_width = left_gripper_width
    next_right_gripper_width = right_gripper_width
    return next_object_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths


def get_transition(object_list, action, meshes, coll_mngr, goal_obj=None, n_planning_trial=10):
    next_object_list = deepcopy(object_list)

    for next_obj in next_object_list:
        coll_mngr.set_transform(next_obj.name, next_obj.pose)

    if action["type"] is "pick":
        pick_obj_idx = get_obj_idx_by_name(next_object_list, action['param'])

        next_object_list[pick_obj_idx].logical_state.clear()
        next_object_list[pick_obj_idx].logical_state["held"] = []
        for next_obj in next_object_list:
            if "prev_pick" in next_obj.logical_state:
                next_obj.logical_state.pop("prev_pick")
        update_logical_state(next_object_list)
        return next_object_list

    if action["type"] is "place":
        place_obj_idx = get_obj_idx_by_name(next_object_list, action['param'])
        held_obj_idx = get_held_object(next_object_list)
        if held_obj_idx is None:
            for obj in next_object_list:
                print(obj.logical_state)
        for _ in range(n_planning_trial):
            if goal_obj is None:
                next_object_pose = sample_on_pose(next_object_list[place_obj_idx], next_object_list[held_obj_idx], meshes,
                                                  coll_mngr)
            else:
                if "custom_table" in action["param"]:
                    next_object_pose = sample_on_pose_with_bias(next_object_list[place_obj_idx], next_object_list[held_obj_idx], goal_obj, meshes,
                                                  coll_mngr, goal_unbias=0.0)
                else:
                    next_object_pose = sample_on_pose_with_bias(next_object_list[place_obj_idx], next_object_list[held_obj_idx], goal_obj, meshes,
                                                  coll_mngr)
            if next_object_pose is None:
                continue

            next_object_list[held_obj_idx].pose = next_object_pose
            next_object_list[held_obj_idx].logical_state.clear()
            next_object_list[held_obj_idx].logical_state["on"] = [next_object_list[place_obj_idx].name]
            next_object_list[held_obj_idx].logical_state["prev_pick"] = []
            update_logical_state(next_object_list)

            return next_object_list
        return None


def height_reward(obj_list, action, next_obj_list):
    if action["type"] is "pick":
        return 0.
    if action["type"] is "place":
        obj_height_list = []
        for obj in obj_list:
            obj_height_list.append(obj.pose[2, 3])
        curr_height = np.max(obj_height_list)

        next_obj_height_list = []
        for next_obj in next_obj_list:
            next_obj_height_list.append(next_obj.pose[2, 3])
        next_height = np.max(next_obj_height_list)
        return next_height - curr_height


def goal_mesh_reward(obj_list, action, next_obj_list, goal_obj, meshes):
    if action["type"] is "pick":
        return 0.
    if action["type"] is "place":
        goal_mesh = deepcopy(meshes[goal_obj.mesh_idx])
        goal_mesh.apply_transform(goal_obj.pose)
        iou = 0.
        for obj in obj_list:
            obj_mesh = deepcopy(meshes[obj.mesh_idx])
            obj_mesh.apply_transform(obj.pose)
            iou += goal_mesh.intersection(obj_mesh).volume/obj_mesh.volume

        next_iou = 0.
        for next_obj in next_obj_list:
            obj_mesh = deepcopy(meshes[next_obj.mesh_idx])
            obj_mesh.apply_transform(next_obj.pose)
            next_iou += goal_mesh.intersection(obj_mesh).volume/obj_mesh.volume

        if next_iou - iou > 0:
            print(next_iou - iou)
        return next_iou - iou


class Tree(object):
    def __init__(self, obj_list, max_depth, coll_mngr, meshes,
                 goal_obj=None,
                 physical_checker=None,
                 min_reeval=10,
                 network=None,
                 robot=None,
                 left_joint_values=None,
                 left_gripper_width=None,
                 right_joint_values=None,
                 right_gripper_width=None):

        self.robot = robot
        if robot is not None:
            self.left_initial_joint_values = left_joint_values
            self.right_initial_joint_values = right_joint_values
            self.get_planning_scene_proxy = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
            self.apply_planning_scene_proxy = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
            self.comput_ik_proxy= rospy.ServiceProxy('/compute_ik', GetPositionIK)
            self.compute_fk_proxy = rospy.ServiceProxy('/compute_fk', GetPositionFK)
            self.mujoco_moveit_planner_proxy = rospy.ServiceProxy('/mujoco_moveit_planner', MuJoCoMoveitConnector)

        self.Tree = nx.DiGraph()
        self.min_reeval = min_reeval
        self.max_depth = max_depth
        self.Tree.add_node(0)

        if goal_obj is None:
            possible_actions = get_possible_actions(obj_list)
        else:
            possible_actions = get_possible_actions_with_bias(obj_list, goal_obj, meshes)
        random.shuffle(possible_actions)
        if robot is not None:
            self.Tree.update(nodes=[(0, {'depth': 0,
                                         'state': obj_list,
                                         'left_joint_values': left_joint_values,
                                         'left_gripper_width': left_gripper_width,
                                         'right_joint_values': right_joint_values,
                                         'right_gripper_width': right_gripper_width,
                                         'reward': 0,
                                         'value': -np.inf,
                                         'visit': 0,
                                         'possible_actions': possible_actions})])
        else:
            self.Tree.update(nodes=[(0, {'depth': 0,
                                         'state': obj_list,
                                         'reward': 0,
                                         'value': -np.inf,
                                         'visit': 0,
                                         'possible_actions': possible_actions})])
        self.coll_mngr = coll_mngr
        self.meshes = meshes
        self.physical_checker = physical_checker
        self.network = network
        self.goal_obj = goal_obj

    def get_reward(self, obj_list, action, next_obj_list):
        if next_obj_list is None:
            return -np.inf
        if self.goal_obj is None:
            return height_reward(obj_list, action, next_obj_list)
        else:
            return goal_mesh_reward(obj_list, action, next_obj_list, self.goal_obj, self.meshes)

    def exploration(self, state_node):
        depth = self.Tree.nodes[state_node]['depth']
        visit = self.Tree.nodes[state_node]['visit']
        self.Tree.update(nodes=[(state_node, {'visit': visit + 1})])

        if depth < self.max_depth:
            obj_list = self.Tree.nodes[state_node]['state']
            if self.robot is not None:
                left_joint_values = self.Tree.nodes[state_node]['left_joint_values']
                left_gripper_width = self.Tree.nodes[state_node]['left_gripper_width']
                right_joint_values = self.Tree.nodes[state_node]['right_joint_values']
                right_gripper_width = self.Tree.nodes[state_node]['right_gripper_width']

            possible_actions = self.Tree.nodes[state_node]['possible_actions']
            action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
            if len(possible_actions) == 0: # No possible actions
                print("no possible actions")
                return 0.0
            elif len(action_nodes) == 0: # There are possible actions, but, they have never been simulated
                print("There are {} possible actions. add one".format(len(possible_actions)))
                print(depth, possible_actions)
                action = possible_actions[0]
                child_action_node = self.Tree.number_of_nodes()
                self.Tree.add_node(child_action_node)
                if self.robot is not None:
                    self.Tree.update(nodes=[(child_action_node,
                                             {'depth': depth,
                                              'state': obj_list,
                                              'left_joint_values': left_joint_values,
                                              'left_gripper_width': left_gripper_width,
                                              'right_joint_values': right_joint_values,
                                              'right_gripper_width': right_gripper_width,
                                              'action': action,
                                              'reward': 0.0,
                                              'value': -np.inf,
                                              'visit': 0})])
                else:
                    self.Tree.update(nodes=[(child_action_node,
                                             {'depth': depth,
                                              'state': obj_list,
                                              'action': action,
                                              'reward': 0.0,
                                              'value': -np.inf,
                                              'visit': 0})])
                self.Tree.add_edge(state_node, child_action_node)
                selected_action_node = child_action_node
            elif len(action_nodes) < len(possible_actions):
                print("There are possible actions. add new action")
                next_action_index = len(action_nodes)
                action=possible_actions[next_action_index]
                child_action_node = self.Tree.number_of_nodes()
                self.Tree.add_node(child_action_node)
                if self.robot is not None:
                    self.Tree.update(nodes=[(child_action_node,
                                             {'depth': depth,
                                              'state': obj_list,
                                              'left_joint_values': left_joint_values,
                                              'left_gripper_width': left_gripper_width,
                                              'right_joint_values': right_joint_values,
                                              'right_gripper_width': right_gripper_width,
                                              'action': action,
                                              'reward': 0.0,
                                              'value': -np.inf,
                                              'visit': 0})])
                else:
                    self.Tree.update(nodes=[(child_action_node,
                                             {'depth': depth,
                                              'state': obj_list,
                                              'action': action,
                                              'reward': 0.0,
                                              'value': -np.inf,
                                              'visit': 0})])
                self.Tree.add_edge(state_node, child_action_node)
                selected_action_node = child_action_node
            else:
                action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
                action_values = [self.Tree.nodes[action_node]['value'] for action_node in action_nodes]

                # exploration method
                action_exploration = 'eps_greedy'
                action_exploration_param = {'eps_min': 0.01}

                if action_exploration == 'random':
                    selected_idx = np.random.choice(len(action_values), size=1)[0]
                elif action_exploration == 'eps_greedy':
                    eps_min = action_exploration_param['eps_min']
                    eps = np.minimum(np.maximum(30.0 / (visit + 1.0), eps_min), 1.0)
                    if eps > np.random.uniform():
                        selected_idx = np.random.choice(len(action_values), size=1)[0]
                    else:
                        selected_idx = np.argmax(action_values)

                selected_action_node = action_nodes[selected_idx]

            self.action_exploration(selected_action_node)
            action_nodes = [action_node for action_node in self.Tree.neighbors(state_node)]
            action_values = [self.Tree.nodes[action_node]['value'] for action_node in action_nodes]
            self.Tree.update(nodes=[(state_node, {'value': np.max(action_values)})])
            return np.max(action_values)
        else:
            return 0.0

    def action_exploration(self, action_node):
        obj_list = self.Tree.nodes[action_node]['state']
        if self.robot is not None:
            left_joint_values = self.Tree.nodes[action_node]['left_joint_values']
            left_gripper_width = self.Tree.nodes[action_node]['left_gripper_width']
            right_joint_values = self.Tree.nodes[action_node]['right_joint_values']
            right_gripper_width = self.Tree.nodes[action_node]['right_gripper_width']
        action = self.Tree.nodes[action_node]['action']
        depth = self.Tree.nodes[action_node]['depth']
        visit = self.Tree.nodes[action_node]['visit']
        self.Tree.update(nodes=[(action_node, {'visit': visit + 1})])

        next_state_nodes = [next_state_node for next_state_node in self.Tree.neighbors(action_node)]
        next_state_visits = [self.Tree.nodes[next_state_node]['visit'] for next_state_node in next_state_nodes]
        next_state_rewards = [self.Tree.nodes[next_state_node]['reward'] for next_state_node in next_state_nodes]

        if len(next_state_nodes) == 0 or np.isinf(next_state_rewards).any() or next_state_visits[-1] >= self.min_reeval / np.maximum(depth, 1):
            print("No next state or all rewards are infinite")
            if self.robot is not None:
                next_obj_list, next_left_arm_joint_values, next_left_gripper_width, next_right_arm_joint_values, \
                next_right_gripper_width, planning_results, planned_trajs, planned_gripper_widths = \
                    get_transition_with_baxter(left_joint_values,
                                               right_joint_values,
                                               self.left_initial_joint_values,
                                               self.right_initial_joint_values,
                                               obj_list,
                                               left_gripper_width,
                                               right_gripper_width,
                                               action,
                                               self.meshes,
                                               self.coll_mngr,
                                               goal_obj=self.goal_obj,
                                               get_planning_scene_proxy=self.get_planning_scene_proxy,
                                               apply_planning_scene_proxy=self.apply_planning_scene_proxy,
                                               compute_fk_proxy=self.compute_fk_proxy,
                                               mujoco_moveit_planner_proxy=self.mujoco_moveit_planner_proxy)
            else:
                next_obj_list = get_transition(obj_list, action, self.meshes, self.coll_mngr, goal_obj=self.goal_obj)

            # Check Feasibility
            if next_obj_list is not None:
                if self.physical_checker is None:
                    is_feasible = True
                else:
                    is_feasible = self.physical_checker(obj_list, action, next_obj_list, self.meshes, network=self.network)

                if not is_feasible:
                    next_obj_list = None
            rew = self.get_reward(obj_list, action, next_obj_list)

            if next_obj_list is None:
                next_obj_list = deepcopy(obj_list)

            if self.goal_obj is None:
                next_possible_actions = get_possible_actions(next_obj_list)
            else:
                next_possible_actions = get_possible_actions_with_bias(next_obj_list, self.goal_obj, self.meshes)
            random.shuffle(next_possible_actions)
            child_node = self.Tree.number_of_nodes()
            self.Tree.add_node(child_node)
            if self.robot is not None:
                self.Tree.update(nodes=[(child_node,
                                         {'depth': depth + 1,
                                          'state': next_obj_list,
                                          'left_joint_values':next_left_arm_joint_values,
                                          'left_gripper_width':next_left_gripper_width,
                                          'right_joint_values':next_right_arm_joint_values,
                                          'right_gripper_width':next_right_gripper_width,
                                          'planning_results':planning_results,
                                          'planned_trajs':planned_trajs,
                                          'planned_gripper_widths':planned_gripper_widths,
                                          'value': -np.inf,
                                          'reward': rew,
                                          'visit': 0,
                                          'possible_actions': next_possible_actions})])
            else:
                self.Tree.update(nodes=[(child_node,
                                         {'depth': depth + 1,
                                          'state': next_obj_list,
                                          'value': -np.inf,
                                          'reward': rew,
                                          'visit': 0,
                                          'possible_actions': next_possible_actions})])
            self.Tree.add_edge(action_node, child_node)
            selected_next_state_node = child_node
        elif next_state_visits[-1] < self.min_reeval / np.maximum(depth, 1):
            print("No next state or all rewards are infinite")
            selected_next_state_node = next_state_nodes[-1]
        else:
            next_state_nodes = [next_state_node for next_state_node in self.Tree.neighbors(action_node)]
            next_state_values = [self.Tree.nodes[next_state_node]['value'] for next_state_node in next_state_nodes]
            selected_next_state_node = next_state_nodes[np.argmax(next_state_values)]

        self.exploration(selected_next_state_node)
        next_state_nodes = [next_state_node for next_state_node in self.Tree.neighbors(action_node)]
        next_state_rewards = [self.Tree.nodes[next_state_node]['reward'] for next_state_node in next_state_nodes]
        next_state_values = [self.Tree.nodes[next_state_node]['value'] for next_state_node in next_state_nodes]
        self.Tree.update(nodes=[(action_node, {'value': np.max(next_state_rewards+next_state_values)})])
        return np.max(next_state_rewards+next_state_values)

    def get_best_path(self, start_node=0):
        next_nodes = [next_node for next_node in self.Tree.neighbors(start_node)]
        if len(next_nodes) == 0:
            return [start_node]
        else:
            best_idx = np.argmax([self.Tree.nodes[next_node]['value'] for next_node in next_nodes])
            next_node = next_nodes[best_idx]
            return [start_node, ] + self.get_best_path(next_node)

    def visualize_tree(self):
        depths = [self.Tree.nodes[n]['depth'] for n in self.Tree.nodes]
        visits = [self.Tree.nodes[n]['visit'] for n in self.Tree.nodes]
        rewards = [self.Tree.nodes[n]['reward'] for n in self.Tree.nodes]
        values = [self.Tree.nodes[n]['value'] for n in self.Tree.nodes]
        actions = [self.Tree.nodes[n]['action']['type']+':'+self.Tree.nodes[n]['action']['param'] if 'action' in self.Tree.nodes[n] else 'state' for n in self.Tree.nodes]
        labels = {
            n: 'depth:{:d}\nvisit:{:d}\nreward:{:.4f}\nvalue:{:.4f}\n{:}'.format(depths[n], visits[n], rewards[n], values[n], actions[n])
            for n in self.Tree.nodes}

        nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        # same layout using matplotlib with no labels
        plt.figure(figsize=(128, 128))
        plt.title('')
        nx.nx_agraph.write_dot(self.Tree, 'test.dot')

        pos = graphviz_layout(self.Tree, prog='dot')
        nx.draw(self.Tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
        plt.show()


def check_stability(obj_idx, object_list, meshes, mesh1):
    if "on" in object_list[obj_idx].logical_state:
        obj_name = object_list[obj_idx].logical_state["on"][0]
        support_obj_idx = get_obj_idx_by_name(object_list, obj_name)
        mesh2 = deepcopy(meshes[object_list[support_obj_idx].mesh_idx])
        mesh2.apply_transform(object_list[support_obj_idx].pose)

        closest_points, dists, surface_idx = trimesh.proximity.closest_point(mesh2, [mesh1.center_mass])
        project_point2 = mesh1.center_mass - closest_points[0]
        project_point2 = project_point2 / np.sqrt(np.sum(project_point2 ** 2.))
        safe_com = project_point2[2] > np.cos(np.pi / 30.)

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


def mujoco_based_physical_checker(obj_list, action, next_obj_list, meshes, network=None):
    if action['type'] is 'pick':
        return True
    if action['type'] is 'place':
        return True


def learning_based_physical_checker(obj_list, action, next_obj_list, meshes, network=None):
    if action['type'] is 'pick':
        return True
    if action['type'] is 'place':
        _, depths, masks = get_image(obj_list, action, next_obj_list, meshes=meshes)
        images = torch.from_numpy(np.asarray([depths])).to('cuda')

        # Forward pass
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        if predicted == 0:
            return False
        else:
            return True


if __name__ == '__main__':
    left_arm_initial_joint_values = {'left_w0': 0.6699952259595108,
                                     'left_w1': 1.030009435085784,
                                     'left_w2': -0.4999997247485215,
                                     'left_e0': -1.189968899785275,
                                     'left_e1': 1.9400238130755056,
                                     'left_s0': -0.08000397926829805,
                                     'left_s1': -0.9999781166910306}

    right_arm_initial_joint_values = {'right_w0': -0.6699952259595108,
                                      'right_w1': 1.030009435085784,
                                      'right_w2': 0.4999997247485215,
                                      'right_e0': 1.189968899785275,
                                      'right_e1': 1.9400238130755056,
                                      'right_s0': -0.08000397926829805,
                                      'right_s1': -0.9999781166910306}

    initial_object_list, meshes, coll_mngr, goal_obj = table_objects_initializer(n_obj_per_mesh_types=[0, 1, 0, 0, 0, 0, 0, 0])
    print('Initialize!!!!!!!!!')
    mcts = Tree(initial_object_list, 2, coll_mngr, meshes,
                goal_obj=goal_obj, physical_checker=None)
    # mcts = Tree(initial_object_list, 10, coll_mngr, meshes,
    #             goal_obj=None, physical_checker=None, robot='baxter',
    #             left_joint_values=left_arm_initial_joint_values,
    #             left_gripper_width=0.03,
    #             right_joint_values=right_arm_initial_joint_values,
    #             right_gripper_width=0.03)

    print('Planning!!!!!!!!!')
    n_exp = 10
    for i in range(n_exp):
        mcts.exploration(0)
        print('{}/{} planning is done'.format(i, n_exp))

    # mcts.visualize_tree()

    opt_path = mcts.get_best_path(0)
    object_list = mcts.Tree.nodes[opt_path[-1]]['state']

    visualize(object_list, meshes)
