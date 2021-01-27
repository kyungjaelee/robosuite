import pickle

from mujoco_py import MjSim, MjViewer

from robosuite.mcts.tree_search import *
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_joint, xml_path_completion

import glfw
import torch

from robosuite.mcts.resnet import ResNet, BasicBlock, Bottleneck
from robosuite.mcts.inception import Inceptionv4


def do_physical_simulation(_sim, _viewer, _object_list, _action, _next_object_list, test_horizon=20000, reset=True):
    state = _sim.get_state()
    if reset:
        for _obj in _object_list:
            if 'custom_table' not in _obj.name:
                pos_arr, quat_arr = T.mat2pose(_obj.pose)
                obj_qpos_addr = _sim.model.get_joint_qpos_addr(_obj.name)
                state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0] + 3] = pos_arr
                state.qpos[obj_qpos_addr[0] + 3:obj_qpos_addr[0] + 7] = quat_arr[[3, 0, 1, 2]]

    if action['type'] == "pick":
        pass
    elif action['type'] == "place":
        held_obj_idx = get_held_object(_object_list)
        _obj = _next_object_list[held_obj_idx]
        pos_arr, quat_arr = T.mat2pose(deepcopy(_obj.pose))
        pos_arr[2] += 0.02
        obj_qpos_addr = _sim.model.get_joint_qpos_addr(_obj.name)
        state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0] + 3] = pos_arr
        state.qpos[obj_qpos_addr[0] + 3:obj_qpos_addr[0] + 7] = quat_arr[[3, 0, 1, 2]]

    _sim.set_state(state)

    for _ in range(test_horizon):
        _sim.step()
        _viewer.render()

    state = _sim.get_state()
    mis_posed_object_list = []
    sim_object_list = []
    for _obj in _next_object_list:
        sim_obj = deepcopy(_obj)
        if 'custom_table' not in _obj.name:
            pos_arr, quat_arr = T.mat2pose(_obj.pose)
            obj_qpos_addr = _sim.model.get_joint_qpos_addr(_obj.name)
            sim_pos_arr = state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0] + 3]
            sim_quat_arr = state.qpos[obj_qpos_addr[0] + 3:obj_qpos_addr[0] + 7]
            sim_obj.pose = T.pose2mat([sim_pos_arr, sim_quat_arr[[1, 2, 3, 0]]])

            dist_diff = np.sqrt(np.sum((pos_arr - sim_pos_arr) ** 2))
            ang_diff = np.arccos(
                np.maximum(np.minimum(2. * np.sum(quat_arr * sim_quat_arr[[1, 2, 3, 0]]) ** 2 - 1., 1.), -1.))
            if dist_diff > 0.05 or ang_diff > np.pi / 20.:
                mis_posed_object_list.append(_obj)

        sim_object_list.append(sim_obj)

    return mis_posed_object_list, sim_object_list


if __name__ == "__main__":

    n_test = 20
    test_early_stop = True
    # model_name_list = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNext50', 'Wide_ResNet50', 'InceptionV4']
    # model_name_list = ['ResNet34', 'ResNet50', 'ResNext50', 'Wide_ResNet50']
    model_name_list = ['Wide_ResNet50']
    for model_name in model_name_list:
        success_list = []
        value_list = []
        path_len_list = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name in 'ResNet18':
            model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)  # ResNet18
        elif model_name in 'ResNet34':
            model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)  # ResNet34
        elif model_name in 'ResNet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)  # ResNet50
        elif model_name in 'ResNext50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4).to(device)  # ResNext50
        elif model_name in 'Wide_ResNet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2).to(device)  # Wide ResNet50
        elif model_name in 'InceptionV4':
            model = Inceptionv4().to(device)

        if test_early_stop:
            model.load_state_dict(torch.load('robosuite/networks/'+model_name+'_depth_img_early_stop.pt'), strict=False)
        else:
            model.load_state_dict(torch.load('robosuite/networks/'+model_name+'_depth_img.pt'), strict=False)

        while len(success_list) < n_test:
            initial_object_list, meshes, coll_mngr, goal_obj = table_objects_initializer()
            print("Initialize!!!")

            mcts = Tree(initial_object_list, 10, coll_mngr, meshes, physical_checker=learning_based_physical_checker, network=model)
            # mcts = Tree(initial_object_list, 10, coll_mngr, meshes, physical_checker=geometry_based_physical_checker, network=None)
            for i in range(30):
                mcts.exploration(0)
                if (i+1)%10 == 0:
                    print('{}th planning is done'.format(i))

            print("Planning!!!")
            optimized_path = mcts.get_best_path(0)
            optimized_rewards = mcts.Tree.nodes[optimized_path[0]]['value']
            print("Optimized rewards : {}".format(optimized_rewards))
            value_list.append(optimized_rewards)

            final_object_list = mcts.Tree.nodes[optimized_path[-1]]['state']
            # visualize(final_object_list, meshes)

            print("Do dynamic simulation!!!")
            if len(success_list) == 0:
                arena_model = MujocoXML(xml_path_completion("arenas/empty_arena.xml"))

                # load baxter model
                baxter_model = MujocoXML(xml_path_completion("robots/baxter/robot.xml"))
                baxter_model.add_prefix("robot0_")

                node = baxter_model.worldbody.find("./body[@name='{}']".format("robot0_base"))
                node.set("pos", array_to_string([-0.66, 0.0, 0.913]))

                # load left gripper
                left_gripper_model = MujocoXML(xml_path_completion("grippers/rethink_gripper.xml"))
                left_gripper_model.add_prefix("gripper0_left_")

                left_arm_subtree = baxter_model.worldbody.find(".//body[@name='robot0_left_hand']")
                for body in left_gripper_model.worldbody:
                    left_arm_subtree.append(body)

                # load right gripper
                right_gripper_model = MujocoXML(xml_path_completion("grippers/rethink_gripper.xml"))
                right_gripper_model.add_prefix("gripper0_right_")

                right_arm_subtree = baxter_model.worldbody.find(".//body[@name='robot0_right_hand']")
                for body in right_gripper_model.worldbody:
                    right_arm_subtree.append(body)

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
                        mj_obj.append(new_joint(name=obj.name, **joint))

                        arena_model.worldbody.append(mj_obj)

                sim = MjSim(arena_model.get_model())
                viewer = MjViewer(sim)

            path_idx = 0
            while path_idx + 2 < len(optimized_path):
                success_flag = True

                state_node = optimized_path[path_idx]
                action_node = optimized_path[path_idx + 1]
                next_state_node = optimized_path[path_idx + 2]

                object_list = mcts.Tree.nodes[state_node]['state']
                action = mcts.Tree.nodes[action_node]['action']
                next_object_list = mcts.Tree.nodes[next_state_node]['state']

                mis_placed_obj_list, sim_object_list = do_physical_simulation(sim, viewer, object_list, action,
                                                                              next_object_list, test_horizon=15000)

                print(action)
                if len(mis_placed_obj_list) > 0:
                    print(len(mis_placed_obj_list))
                    success_flag = False
                else:
                    success_flag = True
                    path_idx += 2

                if not success_flag:
                    break

            success_list.append(success_flag)
            path_len_list.append(path_idx)
            print(len(success_list))
            print(path_idx)

        glfw.destroy_window(viewer.window)
        del sim, viewer

        with open('robosuite/data/mcts_test_result'+'_learning'+model_name+'.pkl', 'wb') as f:
            pickle.dump({'path_len_list': path_len_list,
                         'success_list': success_list,
                         'value_list': value_list}, f, pickle.HIGHEST_PROTOCOL)
