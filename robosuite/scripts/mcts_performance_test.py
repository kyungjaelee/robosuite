import numpy as np
from robosuite.mcts.tree_search_v5 import *

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
    opt_num = 1000
    best_value = -np.inf
    opt_idx = 0
    # visualize(initial_object_list, mcts.meshes, mcts.goal_obj)

    # for _ in range(opt_num):
    #     mcts.exploration(0)
    #     print(mcts.Tree.nodes[0]['value'])

    mcts.exhaustive_search(state_node=0)
    # mcts.visualize()

    best_path_indices = mcts.get_best_path(0)
    best_object_list = mcts.Tree.nodes[best_path_indices[-1]]['state']

    print(mcts.Tree.nodes[0]['value'])

    # visualize(best_object_list, mcts.meshes, mcts.goal_obj)

    mcts.exhaustive_kinematic_search(state_node=0)
    paths, values, kinematic_plans = mcts.get_all_kinematic_path(state_node=0)
    print(paths)
    print(values)
    best_path_indices = paths[-1]
    best_object_list = mcts.Tree.nodes[best_path_indices[-1]]['state']
    visualize(best_object_list, mcts.meshes, mcts.goal_obj)