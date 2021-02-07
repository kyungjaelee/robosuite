import pickle

from mujoco_py import MjSim, MjViewer, MjRenderContextOffscreen

from robosuite.mcts.tree_search import *
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_joint, xml_path_completion

import glfw


def do_physical_simulation(_sim, _object_list, _action, _next_object_list, _viewer=None, test_horizon=20000, reset=True):
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
        if _viewer is not None:
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

    goal_name = 'regular_shapes'
    mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_rotation_types=8)
    for dataset_idx in range(9, 300):
        success_list = []
        configuration_list = []
        action_list = []
        next_configuration_list = []

        print("{}th experiment started".format(dataset_idx))
        while len(configuration_list) < 1e2:
            print("Initialization")
            initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
                configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
                                          goal_name=goal_name)

            mcts = Tree(initial_object_list, np.sum(n_obj_per_mesh_types) * 2, coll_mngr, meshes, contact_points,
                        contact_faces, rotation_types, _goal_obj=goal_obj, _physcial_constraint_checker=None)
            for _ in range(10):
                mcts.exploration(0)


            print("Planning")
            optimized_path = mcts.get_best_path(0)
            final_object_list = mcts.Tree.nodes[optimized_path[-1]]['state']

            print("Do dynamic simulation")
            if len(configuration_list) == 0:
                arena_model = MujocoXML(xml_path_completion("arenas/empty_arena.xml"))
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
                viewer = None
                # viewer = MjViewer(sim)

            path_idx = 0
            while path_idx + 2 < len(optimized_path):
                success_flag = True

                state_node = optimized_path[path_idx]
                action_node = optimized_path[path_idx + 1]
                next_state_node = optimized_path[path_idx + 2]

                object_list = mcts.Tree.nodes[state_node]['state']
                action = mcts.Tree.nodes[action_node]['action']
                next_object_list = mcts.Tree.nodes[next_state_node]['state']

                mis_placed_obj_list, sim_object_list = do_physical_simulation(sim, object_list, action,
                                                                              next_object_list,
                                                                              _viewer=viewer,
                                                                              test_horizon=30000)

                print(action)
                if len(mis_placed_obj_list) > 0:
                    print(len(mis_placed_obj_list))
                    success_flag = False
                else:
                    success_flag = True
                    path_idx += 2

                if action['type'] in 'place':
                    configuration_list.append(object_list)
                    success_list.append(success_flag)
                    action_list.append(action)
                    next_configuration_list.append(next_object_list)

                if not success_flag:
                    break

            print(len(configuration_list))
            print(success_flag)

        if viewer is not None:
            glfw.destroy_window(viewer.window)
        del sim, viewer

        print("{}th experiment ended".format(dataset_idx))
        with open('../data/sim_dynamic_data_'+goal_name+'_'+str(dataset_idx)+'.pkl', 'wb') as f:
            pickle.dump({'configuration_list': configuration_list,
                         'success_list': success_list,
                         'action_list': action_list,
                         'next_configuration_list': next_configuration_list}, f, pickle.HIGHEST_PROTOCOL)
