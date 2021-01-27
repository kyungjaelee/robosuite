from mujoco_py import MjSim, MjViewer, cymj

from robosuite.mcts import *
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

    position_error_right = goal_qpos[qpos_idx_right] - joint_pos_right
    vel_pos_error_right = -joint_vel_right
    desired_torque_right = (np.multiply(np.array(position_error_right), kp*np.ones(len(qpos_idx_right)))
                            + np.multiply(vel_pos_error_right, kd*np.ones(len(qvel_idx_right))))

    # desired torque: left arm
    joint_pos_left = np.array(sim.data.qpos[qpos_idx_left])
    joint_vel_left = np.array(sim.data.qvel[qvel_idx_left])

    position_error_left = goal_qpos[qpos_idx_left] - joint_pos_left
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
        if position_error_right[i] > 0.01:
            goal_reach = False
            break
    for i in range(len(position_error_left)):
        if position_error_left[i] > 0.01:
            goal_reach = False
            break

    return torques, goal_reach


if __name__ == "__main__":
    initial_object_list, meshes, coll_mngr, goal_obj = table_objects_initializer(random_initial=False)

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

    get_planning_scene_proxy = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
    apply_planning_scene_proxy = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
    get_joint_trajectory_proxy = rospy.ServiceProxy('/send_joint_trajectory', JointTrajectory)

    rospy.wait_for_service('/get_planning_scene')
    rospy.wait_for_service('/apply_planning_scene')
    rospy.wait_for_service('/send_joint_trajectory')

    INIT_JOINT_QPOS = np.array([0.0, 0.0, -0.54105206811, 0.0, 0.75049157835, 0.0, 1.25663706144, 0.0, 0.0, 0.0,
                                0.0, -0.54105206811, 0.0, 0.75049157835, 0.0, 1.25663706144, 0.0, 0.0, 0.0])
    TARGET_JOINT_QPOS = np.zeros(19)

    sim = MjSim(arena_model.get_model())
    state = sim.get_state()
    state.qpos[qpos_idx_left] = INIT_JOINT_QPOS[qpos_idx_left]
    state.qpos[qpos_idx_right] = INIT_JOINT_QPOS[qpos_idx_right]
    for obj in initial_object_list:
        if 'custom_table' not in obj.name:
            pos_arr, quat_arr = T.mat2pose(obj.pose)
            obj_qpos_addr = sim.model.get_joint_qpos_addr(obj.name+"_joint")
            state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0]+3] = pos_arr
            state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7] = quat_arr[[3, 0, 1, 2]]
    sim.set_state(state)
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0

    for _ in range(10000):
        torques, is_reach = joint_position_control(sim, INIT_JOINT_QPOS)
        sim.data.ctrl[:14] = torques

        sim.step()
        viewer.render()

    qpos_trajectory = planning_with_target_pose(sim, TARGET_JOINT_QPOS,
                                                get_planning_scene_proxy,
                                                apply_planning_scene_proxy,
                                                get_joint_trajectory_proxy)

    for i in range(len(qpos_trajectory)):
        while True:
            torques, is_reach = joint_position_control(sim, qpos_trajectory[i])
            sim.data.ctrl[:14] = torques
            sim.step()
            viewer.render()
            if is_reach:
                print("reach the {}-th goal pos".format(i + 1))
                break
