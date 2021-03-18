import pickle
import glfw

from mujoco_py import MjSim, MjViewer

from robosuite.mcts.tree_search_v2 import *
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_joint, xml_path_completion

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_trimesh(trimesh_data, transforms=np.eye(4, 4), ax=None, color=None):
    if ax is None:
        ax = plt.gca(projection='3d')
    if color is None:
        color = [np.random.uniform(), np.random.uniform(), np.random.uniform(), 0.1]

    transformed_vertices = (transforms[:3, :3].dot(trimesh_data.vertices.T) + transforms[:3, [3]]).T

    ax.plot_trisurf(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2],
                    triangles=trimesh_data.faces,
                    linewidth=0.2, antialiased=True, color=color, edgecolor=None)

    # axes = (0.05 * transforms[:3, :3] + transforms[:3, [3]]).T
    # ax.plot([transforms[0, 3], axes[0, 0]], [transforms[1, 3], axes[0, 1]], [transforms[2, 3], axes[0, 2]], c='r')
    # ax.plot([transforms[0, 3], axes[1, 0]], [transforms[1, 3], axes[1, 1]], [transforms[2, 3], axes[1, 2]], c='g')
    # ax.plot([transforms[0, 3], axes[2, 0]], [transforms[1, 3], axes[2, 1]], [transforms[2, 3], axes[2, 2]], c='b')
    return ax


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
    goal_name_list = ['stack_easy', 'stack_hard', 'tower_goal', 'twin_tower_goal']
    area_ths_list = [1., 1., 0.003, 0.003]
    exploration_list = [{'method': 'bai_perturb', 'param': 1e-1},
                        {'method': 'bai_perturb', 'param': 1.},
                        {'method': 'bai_perturb', 'param': 10.},
                        {'method': 'bai_perturb', 'param': 100.},
                        {'method': 'ucb', 'param': 1e-1},
                        {'method': 'ucb', 'param': 1.},
                        {'method': 'ucb', 'param': 10.},
                        {'method': 'ucb', 'param': 100.},
                        {'method': 'bai_ucb', 'param': 1e-1},
                        {'method': 'bai_ucb', 'param': 1.},
                        {'method': 'bai_ucb', 'param': 10.},
                        {'method': 'bai_ucb', 'param': 100.},
                        {'method': 'random', 'param': 'random'}]

    for goal_name, area_ths in zip(goal_name_list, area_ths_list):
        mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes(_area_ths=area_ths)
        initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, n_obj_per_mesh_types = \
            configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points,
                                      goal_name=goal_name)

        print('Problem : {}'.format(goal_name))
        n_seed = 10
        opt_num = 300

        for exploration in exploration_list:
            print("Exploration method : ", exploration)
            seed_value_list = []
            seed_value_indices = []
            seed_final_state_list = []
            seed_path_length_list = []

            for seed in range(n_seed):
                np.random.seed(seed)
                mcts = Tree(initial_object_list, np.sum(n_obj_per_mesh_types) * 2, coll_mngr, meshes, contact_points,
                            contact_faces, rotation_types,
                            _goal_obj=goal_obj,
                            _exploration=exploration)
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

                optimized_path = mcts.get_best_path()
                object_list = mcts.Tree.nodes[optimized_path[-1]]['state']
                # ax = None
                # for obj in object_list:
                #     ax = visualize_trimesh(meshes[obj.mesh_idx], obj.pose, ax=ax, color=obj.color)
                # if goal_obj is not None:
                #     ax = visualize_trimesh(meshes[goal_obj.mesh_idx], goal_obj.pose, ax=ax, color=goal_obj.color)
                # plt.show()

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

                if goal_obj is not None:
                    mj_goal_obj_model = MujocoXML(xml_path_completion("objects/"+goal_obj.name+".xml"))
                    goal_body = mj_goal_obj_model.worldbody.find("./body[@name='"+goal_obj.name+"']")
                    goal_pos_arr, goal_quat_arr = T.mat2pose(goal_obj.pose)
                    goal_body.set("pos", array_to_string(goal_pos_arr))
                    goal_body.set("quat", array_to_string(goal_quat_arr[[3, 0, 1, 2]]))
                    arena_model.merge(mj_goal_obj_model)

                sim = MjSim(arena_model.get_model())
                # viewer = MjViewer(sim)

                path_idx = 0
                while path_idx + 2 <= len(optimized_path):
                    state_node = optimized_path[path_idx]
                    action_node = optimized_path[path_idx + 1]
                    next_state_node = optimized_path[path_idx + 2]

                    object_list = mcts.Tree.nodes[state_node]['state']
                    action = mcts.Tree.nodes[action_node]['action']
                    next_object_list = mcts.Tree.nodes[next_state_node]['state']

                    mis_placed_obj_list, sim_object_list = do_physical_simulation(sim, object_list, action,
                                                                                  next_object_list,
                                                                                  # _viewer=viewer,
                                                                                  test_horizon=30000)

                    print(seed, action['type'], action['param'], path_idx, len(optimized_path))
                    if len(mis_placed_obj_list) > 0:
                        print('Fail!!')
                        break

                    path_idx += 2

                seed_value_list.append(best_value_list)
                seed_value_indices.append(best_value_indices)
                seed_final_state_list.append(best_final_state_list)
                seed_path_length_list.append(path_idx)

                # glfw.destroy_window(viewer.window)
                del sim

            with open('../data/mcts_performance_' + goal_name + '_' + exploration['method'] + '_' + str(exploration['param']) + '.pkl', 'wb') as f:
                pickle.dump({'seed_value_list': seed_value_list,
                             'seed_value_indices': seed_value_indices,
                             'seed_final_state_list': seed_final_state_list,
                             'seed_path_length_list': seed_path_length_list}, f, pickle.HIGHEST_PROTOCOL)

