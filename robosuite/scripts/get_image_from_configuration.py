import pickle
import numpy as np

from robosuite.mcts.util import *
from robosuite.mcts.util import get_image
from robosuite.mcts.util import get_meshes


def get_supporting_object(obj_idx, object_list):
    naming_list = [object_list[obj_idx].name]
    if "on" in object_list[obj_idx].logical_state:
        for obj_name in object_list[obj_idx].logical_state["on"]:
            support_obj_idx = get_obj_idx_by_name(object_list, obj_name)
            naming_list += get_supporting_object(support_obj_idx, object_list)
    return naming_list


if __name__ == "__main__":

    color1_list = []
    color2_list = []
    depth1_list = []
    depth2_list = []
    mask1_list = []
    mask2_list = []
    label_list = []

    batch_size = 128
    batch_num = 0
    total_success_rate = 0
    goal_name = 'round_shapes'
    _, _, _, meshes, _, _, _ = get_meshes(_area_ths=1.)
    round_shape_list = ['arch_box', 'half_cylinder_box', 'triangle_box']

    for dataset_idx in range(346):
        # with open('../data/sim_dynamic_data_'+goal_name+'_'+str(dataset_idx)+'.pkl', 'rb') as f:
        with open('../prev_data/sim_dynamic_data'+str(dataset_idx)+'.pkl', 'rb') as f:
            data = pickle.load(f)

        configuration_list = data['configuration_list']
        success_list = data['success_list']
        action_list = data['action_list']
        next_configuration_list = data['next_configuration_list']
        total_success_rate += np.mean(success_list) / 346
        print('{}/{} success rate : {}'.format(dataset_idx+1, 346, np.mean(success_list)))

        for object_list, action, next_object_list, label in zip(configuration_list, action_list, next_configuration_list, success_list):
            place_idx = get_obj_idx_by_name(next_object_list, action['param'])
            pick_idx = get_held_object(object_list)
            tower_name_list = get_supporting_object(place_idx, object_list)
            tower_name_list.append(object_list[pick_idx].name)
            # print(tower_name_list)
            # print(len([1 for shape_name in round_shape_list for tower_name in tower_name_list if shape_name in tower_name]))
            if len([1 for shape_name in round_shape_list for tower_name in tower_name_list if shape_name in tower_name]) > 0:
                colors, depths, masks = get_image(object_list, action, next_object_list, meshes=meshes, label=label, do_visualize=False)
                color1_list.append([colors[0]])
                color2_list.append([colors[1]])
                depth1_list.append([depths[0]])
                depth2_list.append([depths[1]])
                mask1_list.append(masks[0])
                mask2_list.append(masks[1])
                label_list.append(label)

            if batch_size == len(label_list):
                with open('../prev_data/img_data_test'+str(batch_num)+'.pkl', 'wb') as f:
                    pickle.dump({'color1_list': color1_list,
                                 'color2_list': color2_list,
                                 'depth1_list': depth1_list,
                                 'depth2_list': depth2_list,
                                 'mask1_list': mask1_list,
                                 'mask2_list': mask2_list,
                                 'label': label_list}, f, pickle.HIGHEST_PROTOCOL)

                    print('{} batch is collected'.format(batch_num))
                    color1_list = []
                    color2_list = []
                    depth1_list = []
                    depth2_list = []
                    mask1_list = []
                    mask2_list = []
                    label_list = []
                    batch_num += 1
    print('Total rate : {}'.format(total_success_rate))
