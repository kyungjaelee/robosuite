import robosuite as suite

from robosuite.mcts.util import *
from robosuite.controllers import load_controller_config

if __name__ == "__main__":

    initial_object_list, meshes, coll_mngr = table_objects_initializer()

    # Create dict to hold options that will be passed to env creation call
    options = {"env_name": 'BaxterStack', "env_configuration": 'bimanual', "robots": 'Baxter',
               "controller_configs": load_controller_config(default_controller='JOINT_POSITION')}

    # initialize the task
    env = suite.make(
        **options,
        initial_object_list=initial_object_list,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.render()

    object_list = initial_object_list
    action_list = get_possible_actions(object_list)
    action = action_list[0]
    print(action)

    if action["type"] is "pick":
        pick_obj_idx = get_obj_idx_by_name(object_list, action['param'])

        object_list[pick_obj_idx].logical_state.clear()
        object_list[pick_obj_idx].logical_state["held"] = []
        update_logical_state(object_list)

        for obj in object_list:
            print(obj.name, obj.logical_state)

    action_list = get_possible_actions(object_list)
    action = action_list[0]
    print(action)
    # Get action limits
    low, high = env.action_spec

    if action["type"] is "place":
        place_obj_idx = get_obj_idx_by_name(object_list, action['param'])
        held_obj_idx = get_held_object(object_list)
        object_list[held_obj_idx].pose = sample_on_pose(object_list[place_obj_idx], object_list[held_obj_idx], meshes,
                                                        coll_mngr)

        object_list[held_obj_idx].logical_state.clear()
        object_list[held_obj_idx].logical_state["on"] = [object_list[place_obj_idx].name]
        update_logical_state(object_list)

        env.action_object(object_list)
        for _ in range(500):
            obs, reward, done, _ = env.step(np.zeros((env.action_dim,)))
            env.render()
        for obj in object_list:
            print(obj.name, obj.logical_state)

    for _ in range(1000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(np.zeros((env.action_dim,)))
        env.render()