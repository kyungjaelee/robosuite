from mujoco_py import MjSim, MjViewer, cymj

from robosuite.mcts.tree_search_v5 import *
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_joint, xml_path_completion

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rospy
from moveit_msgs.srv import *
from mujoco_moveit_connector.srv import *


def visualize_trimesh(trimesh_data, transforms=np.eye(4, 4), ax=None, color=None):
    if ax is None:
        ax = plt.gca(projection='3d')
    if color is None:
        color = [np.random.uniform(), np.random.uniform(), np.random.uniform(), 0.2]

    transformed_vertices = (transforms[:3, :3].dot(trimesh_data.vertices.T) + transforms[:3, [3]]).T

    ax.plot_trisurf(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2],
                    triangles=trimesh_data.faces,
                    linewidth=0.2, antialiased=True, color=color, edgecolor='gray')

    axes = (0.05 * transforms[:3, :3] + transforms[:3, [3]]).T
    ax.plot([transforms[0, 3], axes[0, 0]], [transforms[1, 3], axes[0, 1]], [transforms[2, 3], axes[0, 2]], c='r')
    ax.plot([transforms[0, 3], axes[1, 0]], [transforms[1, 3], axes[1, 1]], [transforms[2, 3], axes[1, 2]], c='g')
    ax.plot([transforms[0, 3], axes[2, 0]], [transforms[1, 3], axes[2, 1]], [transforms[2, 3], axes[2, 2]], c='b')
    return ax


kp = 50.
kd = 14.14213562

qpos_idx_right = [1, 2, 3, 4, 5, 6, 7]
qvel_idx_right = [1, 2, 3, 4, 5, 6, 7]
qpos_idx_left = [10, 11, 12, 13, 14, 15, 16]
qvel_idx_left = [10, 11, 12, 13, 14, 15, 16]

torques_high = [50., 100., 50., 50., 15., 15., 15., 50., 100., 50., 50., 15., 15., 15.]
torques_low = [-50., -100., -50., -50., -15., -15., -15., -50., -100., -50., -50., -15., -15., -15.]

ARM_JOINT_NAME = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2',
                  'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']


def planning_with_target_pose(sim, TARGET_JOINT_QPOS, get_planning_scene_proxy, apply_planning_scene_proxy, get_joint_trajectory_proxy):
    resp = get_planning_scene_proxy(GetPlanningSceneRequest())
    current_scene = resp.scene

    next_scene = deepcopy(current_scene)
    next_scene.robot_state.joint_state.name = ARM_JOINT_NAME
    next_scene.robot_state.joint_state.position = np.concatenate((sim.data.qpos[1:8], sim.data.qpos[10:17]), axis=-1)
    next_scene.robot_state.joint_state.velocity = [0]*len(ARM_JOINT_NAME)
    next_scene.robot_state.joint_state.effort = [0]*len(ARM_JOINT_NAME)
    next_scene.robot_state.is_diff = True

    next_scene.is_diff = True
    req = ApplyPlanningSceneRequest()
    req.scene = next_scene
    resp = apply_planning_scene_proxy(req)
    for _ in range(100):
        rospy.sleep(0.001)

    if resp.success:
        # get trajectory from moveit
        joint_trajectory = get_joint_trajectory_proxy(ARM_JOINT_NAME, TARGET_JOINT_QPOS)
        for _ in range(100):
            rospy.sleep(0.001)
        qpos_trajectory = np.array(
            [np.concatenate(([0.0], p.positions[7:], [0.0, 0.0], p.positions[:7], [0.0, 0.0])) for p in joint_trajectory.plan.points])
        return qpos_trajectory
    else:
        return None


def joint_position_control(sim, goal_qpos):
    # desired torque: right arm
    joint_pos_right = np.array(sim.data.qpos[qpos_idx_right])
    joint_vel_right = np.array(sim.data.qvel[qvel_idx_right])

    position_error_right = goal_qpos[:7] - joint_pos_right
    vel_pos_error_right = -joint_vel_right
    desired_torque_right = (np.multiply(np.array(position_error_right), kp*np.ones(len(qpos_idx_right)))
                            + np.multiply(vel_pos_error_right, kd*np.ones(len(qvel_idx_right))))

    # desired torque: left arm
    joint_pos_left = np.array(sim.data.qpos[qpos_idx_left])
    joint_vel_left = np.array(sim.data.qvel[qvel_idx_left])

    position_error_left = goal_qpos[7:] - joint_pos_left
    vel_pos_error_left = -joint_vel_left
    desired_torque_left = (np.multiply(np.array(position_error_left), kp*np.ones(len(qpos_idx_left)))\
                           + np.multiply(vel_pos_error_left, kd*np.ones(len(qvel_idx_left))))

    # calculate mass-matrix
    mass_matrix = np.ndarray(shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order='C')
    cymj._mj_fullM(sim.model, mass_matrix, sim.data.qM)
    mass_matrix = np.reshape(mass_matrix, (len(sim.data.qvel), len(sim.data.qvel)))
    mass_matrix_right = mass_matrix[qvel_idx_right, :][:, qvel_idx_right]
    mass_matrix_left = mass_matrix[qvel_idx_left, :][:, qvel_idx_left]

    # calculate torque-compensation
    torque_compensation_right = sim.data.qfrc_bias[qvel_idx_right]
    torque_compensation_left = sim.data.qfrc_bias[qvel_idx_left]

    # calculate torque values
    torques_right = np.dot(mass_matrix_right, desired_torque_right) + torque_compensation_right
    torques_left = np.dot(mass_matrix_left, desired_torque_left) + torque_compensation_left

    torques = np.concatenate((torques_right, torques_left))
    torques = np.clip(torques, torques_low, torques_high)

    goal_reach = True
    for i in range(len(position_error_right)):
        if np.abs(position_error_right[i]) > 0.005:
            goal_reach = False
            break
    for i in range(len(position_error_left)):
        if np.abs(position_error_left[i]) > 0.005:
            goal_reach = False
            break

    return torques, goal_reach


if __name__ == "__main__":
    np.random.seed(1)
    mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=0.003)
    initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
        configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
                                  goal_name='tower_goal')

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
                _exploration={'method': 'random', 'param': 1.},
                _get_planning_scene_proxy=get_planning_scene_proxy,
                _apply_planning_scene_proxy=apply_planning_scene_proxy,
                _compute_fk_proxy=compute_fk_proxy,
                _cartesian_planning_with_gripper_pose_proxy=cartesian_planning_with_gripper_pose_proxy,
                _planning_with_gripper_pose_proxy=planning_with_gripper_pose_proxy,
                _planning_with_arm_joints_proxy=planning_with_arm_joints_proxy)
    opt_num = 30
    best_value = -np.inf
    opt_idx = 0

    mcts.exhaustive_search(state_node=0)
    mcts.visualize()

    best_path_indices = mcts.get_best_path(0)
    best_object_list = mcts.Tree.nodes[best_path_indices[-1]]['state']

    print(mcts.Tree.nodes[0]['value'])
    print(best_path_indices)

    visualize(best_object_list, mcts.meshes, mcts.goal_obj)

    mcts.exhaustive_kinematic_search(state_node=0)
    paths, values, kinematic_plans = mcts.get_all_kinematic_path(state_node=0)

    print(paths)
    print(values)

    # Do dynamic simulation in MuJoCo
    arena_model = MujocoXML(xml_path_completion("arenas/empty_arena.xml"))

    # load baxter model
    baxter_model = MujocoXML(xml_path_completion("robots/baxter/robot.xml"))
    baxter_model.add_prefix("robot0_")

    node = baxter_model.worldbody.find("./body[@name='{}']".format("robot0_base"))
    node.set("pos", array_to_string([-0.0, 0.0, 0.913]))

    # load left gripper
    left_gripper_model = MujocoXML(xml_path_completion("grippers/rethink_gripper.xml"))
    left_gripper_model.add_prefix("gripper0_left_")

    left_arm_subtree = baxter_model.worldbody.find(".//body[@name='robot0_left_hand']")
    for body in left_gripper_model.worldbody:
        left_arm_subtree.append(body)

    site = left_gripper_model.worldbody.find(".//site[@name='{}']".format('gripper0_left_grip_site'))
    site.set("rgba", "0 0 0 0")
    site = left_gripper_model.worldbody.find(".//site[@name='{}']".format('gripper0_left_grip_site_cylinder'))
    site.set("rgba", "0 0 0 0")

    # load right gripper
    right_gripper_model = MujocoXML(xml_path_completion("grippers/rethink_gripper.xml"))
    right_gripper_model.add_prefix("gripper0_right_")

    right_arm_subtree = baxter_model.worldbody.find(".//body[@name='robot0_right_hand']")
    for body in right_gripper_model.worldbody:
        right_arm_subtree.append(body)

    site = right_gripper_model.worldbody.find(".//site[@name='{}']".format('gripper0_right_grip_site'))
    site.set("rgba", "0 0 0 0")
    site = right_gripper_model.worldbody.find(".//site[@name='{}']".format('gripper0_right_grip_site_cylinder'))
    site.set("rgba", "0 0 0 0")

    # merge XML
    baxter_model.merge(left_gripper_model, merge_body=False)
    baxter_model.merge(right_gripper_model, merge_body=False)
    arena_model.merge(baxter_model)

    for obj in initial_object_list:
        if 'gripper' not in obj.name:
            if 'custom_table' in obj.name:
                table_model = MujocoXML(xml_path_completion("objects/custom_table.xml"))
                table_pos_arr, table_quat_arr = T.mat2pose(obj.pose)
                table_body = table_model.worldbody.find("./body[@name='custom_table']")
                table_body.set("pos", array_to_string(table_pos_arr))
                table_body.set("quat", array_to_string(table_quat_arr[[3, 0, 1, 2]]))
                arena_model.merge(table_model)
            else:
                if 'arch_box' in obj.name:
                    obj_xml_path = "objects/arch_box.xml"
                elif 'rect_box' in obj.name:
                    obj_xml_path = "objects/rect_box.xml"
                elif 'square_box' in obj.name:
                    obj_xml_path = "objects/square_box.xml"
                elif 'half_cylinder_box' in obj.name:
                    obj_xml_path = "objects/half_cylinder_box.xml"
                elif 'triangle_box' in obj.name:
                    obj_xml_path = "objects/triangle_box.xml"

                mj_obj_model = MujocoXMLObject(xml_path_completion(obj_xml_path), name=obj.name)

                arena_model.merge_asset(mj_obj_model)
                mj_obj = mj_obj_model.get_collision(site=True)

                joint = mj_obj_model.joints[0]
                mj_obj.append(new_joint(name=obj.name+"_joint", **joint))

                arena_model.worldbody.append(mj_obj)

    init_left_joint_values = mcts.Tree.nodes[0]['left_joint_values']
    init_right_joint_values = mcts.Tree.nodes[0]['right_joint_values']

    sim = MjSim(arena_model.get_model())
    state = sim.get_state()
    state.qpos[qpos_idx_left] = [init_left_joint_values[left_joint_name] for left_joint_name in ARM_JOINT_NAME[7:]]
    state.qpos[qpos_idx_right] = [init_right_joint_values[left_joint_name] for left_joint_name in ARM_JOINT_NAME[:7]]
    for obj in initial_object_list:
        if 'gripper' not in obj.name and 'custom_table' not in obj.name:
            pos_arr, quat_arr = T.mat2pose(obj.pose)
            obj_qpos_addr = sim.model.get_joint_qpos_addr(obj.name+"_joint")
            state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0]+3] = pos_arr
            state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7] = quat_arr[[3, 0, 1, 2]]
    sim.set_state(state)
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0
    init_pose = np.array([init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + [init_left_joint_values[left_joint_name] for left_joint_name in ARM_JOINT_NAME[7:]])
    path_idx = 0
    ctrl_timeout = 30000

    while path_idx < len(best_path_indices)-2 and len(best_path_indices) > 1:
        state_node = best_path_indices[path_idx]
        action_node = best_path_indices[path_idx+1]
        next_state_node = best_path_indices[path_idx+2]

        action = mcts.Tree.nodes[action_node]['action']
        depth = mcts.Tree.nodes[next_state_node]['depth']
        print("current depth:", depth)
        if "pick" in action['type']:
            pick_pre_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][0]
            pick_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][1]
            pick_retreat_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][2]
            pick_pre_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in pick_pre_traj.points])
            pick_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in pick_traj.points])
            pick_retreat_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in pick_retreat_traj.points])
            print("======================Go to pregrasp pose=========================")
            for i in range(len(pick_pre_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, pick_pre_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [-0.0115, 0.0115]
                    sim.step()
                    viewer.render()

            print("======================Pick=========================")
            for i in range(len(pick_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, pick_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [-0.0115, 0.0115]
                    sim.step()
                    viewer.render()

            for _ in range(1000):
                sim.data.ctrl[14:16] = [0.020833, -0.020833]
                sim.step()
                viewer.render()
            print("gripper is closed")

            print("======================Retreat to initial pose=========================")
            for i in range(len(pick_retreat_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, pick_retreat_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [0.020833, -0.020833]
                    sim.step()
                    viewer.render()
        elif "place" in action["type"]:
            plcae_approach_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][0]
            plcae_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][1]
            plcae_safe_retreat_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][2]
            plcae_retreat_traj = mcts.Tree.nodes[next_state_node]['planned_traj_list'][3]
            place_approach_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in plcae_approach_traj.points])
            place_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in plcae_traj.points])
            plcae_safe_retreat_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in plcae_safe_retreat_traj.points])
            plcae_retreat_qpos_traj = np.array([[init_right_joint_values[right_joint_name] for right_joint_name in ARM_JOINT_NAME[:7]] + list(p.positions) for p in plcae_retreat_traj.points])

            print("======================Go to preplace pose=========================")
            for i in range(len(place_approach_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, place_approach_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [0.020833, -0.020833]
                    sim.step()
                    viewer.render()

            print("======================Place=========================")
            for i in range(len(place_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, place_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [0.020833, -0.020833]
                    sim.step()
                    viewer.render()

            for _ in range(1000):
                sim.data.ctrl[14:16] = [-0.0115, 0.0115]
                sim.step()
                viewer.render()
            print("gripper is open")

            print("======================Retreat to preplace pose=========================")
            for i in range(len(plcae_safe_retreat_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, plcae_safe_retreat_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [-0.0115, 0.0115]
                    sim.step()
                    viewer.render()

            print("======================Retreat to initial pose=========================")
            for i in range(len(plcae_retreat_qpos_traj)):
                for _ in range(ctrl_timeout):
                    torques, is_reach = joint_position_control(sim, plcae_retreat_qpos_traj[i])
                    if is_reach:
                        print("reach the {}-th goal pos".format(i + 1))
                        break

                    sim.data.ctrl[:14] = torques
                    sim.data.ctrl[14:16] = [-0.0115, 0.0115]
                    sim.step()
                    viewer.render()
        print("I am here")
        path_idx += 2
    print("finish all trajectories")
    print("staying")
    while True:
        torques, is_reach = joint_position_control(sim, init_pose)
        sim.data.ctrl[:14] = torques
        sim.step()
        viewer.render()