from collections import OrderedDict
from copy import deepcopy
import numpy as np

from robosuite.environments.robot_env import RobotEnv

from robosuite.models.arenas import CustomTableArena
from robosuite.models.base import MujocoXML
from robosuite.models.objects import TriangleBoxObject, SquareBoxObject, ArchBoxObject, RectangleBoxObject, HalfCylinderBoxObject, CanObject
from robosuite.models.world import MujocoWorldBase
from robosuite.models.robots import check_bimanual

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import xml_path_completion, new_joint, array_to_string


class BaxterStack(RobotEnv):
    """
    This class corresponds to the lifting task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table (Default option)

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler instance): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        use_indicator_object (bool): if True, sets up an indicator object that
            is useful for debugging.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="single-arm-opposed",
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        initial_object_list=None,
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # First, verify that correct number of robots are being inputted
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        self.initial_object_list = []
        for obj in initial_object_list:
            if 'custom_table' in obj.name:
                self.table_object = obj
            else:
                self.initial_object_list.append(obj)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        return 0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        self.robots[0].robot_model.set_base_xpos([0., 0., 0.])

        # load model for table top workspace
        self.mujoco_arena = CustomTableArena()
        table_pos_arr, table_quat_arr = T.mat2pose(self.table_object.pose)

        custom_table = self.mujoco_arena.worldbody.find("./body[@name='custom_table']")
        custom_table.set("pos", array_to_string(table_pos_arr))
        custom_table.set("quat", array_to_string(table_quat_arr))

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        self.mujoco_objects = OrderedDict()
        for obj in self.initial_object_list:
            if 'arch_box' in obj.name:
                self.mujoco_objects.update({obj.name: ArchBoxObject()})
            elif 'rect_box' in obj.name:
                self.mujoco_objects.update({obj.name: RectangleBoxObject()})
            elif 'square_box' in obj.name:
                self.mujoco_objects.update({obj.name: SquareBoxObject()})
            elif 'half_cylinder_box' in obj.name:
                self.mujoco_objects.update({obj.name: HalfCylinderBoxObject()})
            elif 'triangle_box' in obj.name:
                self.mujoco_objects.update({obj.name: TriangleBoxObject()})

        # task includes arena, robot, and objects of interest
        self.model = MujocoWorldBase()
        self.model.merge(self.mujoco_arena)
        for robot in self.robots:
            self.model.merge(robot.robot_model)

        self.model.objects = []
        self.model.max_horizontal_radius = 0

        # xml manifestations of all objects
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            self.model.merge_asset(obj_mjcf)
            obj = obj_mjcf.get_collision(site=True)

            for i, joint in enumerate(obj_mjcf.joints):
                obj.append(new_joint(name="{}_jnt{}".format(obj_name, i), **joint))
            self.model.objects.append(obj)
            self.model.worldbody.append(obj)
            self.model.max_horizontal_radius = max(
                self.model.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

        for i in range(len(self.model.objects)):
            pose = deepcopy(self.initial_object_list[i].pose)
            pos_arr, quat_arr = T.mat2pose(pose)
            pos_arr[2] += 0.00

            self.model.objects[i].set("pos", array_to_string(pos_arr))
            self.model.objects[i].set("quat", array_to_string(quat_arr))

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simu
        lation data.
        """
        super()._get_reference()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def action_object(self, object_list_w_table):

        object_list = []
        for obj in object_list_w_table:
            if 'custom_table' not in obj.name:
                object_list.append(obj)

        print(type(self.sim))

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            if self.env_configuration == "bimanual":
                pr0 = self.robots[0].robot_model.naming_prefix + "left_"
                pr1 = self.robots[0].robot_model.naming_prefix + "right_"
            else:
                pr0 = self.robots[0].robot_model.naming_prefix
                pr1 = self.robots[1].robot_model.naming_prefix

            # position and rotation of object
            cube_pos = np.array(self.sim.data.body_xpos[self.pot_body_id])
            cube_quat = T.convert_quat(
                self.sim.data.body_xquat[self.pot_body_id], to="xyzw"
            )
            di["cube_pos"] = cube_pos
            di["cube_quat"] = cube_quat

            di[pr0 + "eef_xpos"] = self._eef0_xpos
            di[pr1 + "eef_xpos"] = self._eef1_xpos
            di["handle_0_xpos"] = np.array(self._handle_0_xpos)
            di["handle_1_xpos"] = np.array(self._handle_1_xpos)
            di[pr0 + "gripper_to_handle"] = np.array(self._gripper_0_to_handle)
            di[pr1 + "gripper_to_handle"] = np.array(self._gripper_1_to_handle)

            di["object-state"] = np.concatenate(
                [
                    di["cube_pos"],
                    di["cube_quat"],
                    di[pr0 + "eef_xpos"],
                    di[pr1 + "eef_xpos"],
                    di["handle_0_xpos"],
                    di["handle_1_xpos"],
                    di[pr0 + "gripper_to_handle"],
                    di[pr1 + "gripper_to_handle"],
                ]
            )

        return di

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.get_top_offset()[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        return pot_bottom_height > table_height + 0.10

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        robots = robots if type(robots) == list or type(robots) == tuple else [robots]
        if self.env_configuration == "single-arm-opposed" or self.env_configuration == "single-arm-parallel":
            # Specifically two robots should be inputted!
            is_bimanual = False
            if type(robots) is not list or len(robots) != 2:
                raise ValueError("Error: Exactly two single-armed robots should be inputted "
                                 "for this task configuration!")
        elif self.env_configuration == "bimanual":
            is_bimanual = True
            # Specifically one robot should be inputted!
            if type(robots) is list and len(robots) != 1:
                raise ValueError("Error: Exactly one bimanual robot should be inputted "
                                 "for this task configuration!")
        else:
            # This is an unknown env configuration, print error
            raise ValueError("Error: Unknown environment configuration received. Only 'bimanual',"
                             "'single-arm-parallel', and 'single-arm-opposed' are supported. Got: {}"
                             .format(self.env_configuration))

        # Lastly, check to make sure all inputted robot names are of their correct type (bimanual / not bimanual)
        for robot in robots:
            if check_bimanual(robot) != is_bimanual:
                raise ValueError("Error: For {} configuration, expected bimanual check to return {}; "
                                 "instead, got {}.".format(self.env_configuration, is_bimanual, check_bimanual(robot)))

    @property
    def _handle_0_xpos(self):
        """
        Grab the position of the left (blue) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle_0_site_id]

    @property
    def _handle_1_xpos(self):
        """
        Grab the position of the right (green) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle_1_site_id]

    @property
    def _pot_quat(self):
        """
        Grab the orientation of the pot body.

        Returns:
            np.array: (x,y,z,w) quaternion of the pot body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

    @property
    def _world_quat(self):
        """
        Grab the world orientation

        Returns:
            np.array: (x,y,z,w) world quaternion
        """
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _eef0_xpos(self):
        """
        Grab the position of Robot 0's end effector.

        Returns:
            np.array: (x,y,z) position of EEF0
        """
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["left"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef1_xpos(self):
        """
        Grab the position of Robot 1's end effector.

        Returns:
            np.array: (x,y,z) position of EEF1
        """
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id])

    @property
    def _gripper_0_to_handle(self):
        """
        Calculate vector from the left gripper to the left pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._handle_0_xpos - self._eef0_xpos

    @property
    def _gripper_1_to_handle(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._handle_1_xpos - self._eef1_xpos
