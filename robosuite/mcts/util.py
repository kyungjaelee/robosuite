import math
from copy import deepcopy
import numpy as np

import trimesh
import pyrender

try:
    import rospy

    # ROS Fundamental packages
    from geometry_msgs.msg import *
    from sensor_msgs.msg import *
    from std_msgs.msg import *
    from shape_msgs.msg import *
    from control_msgs.msg import *

    # ROS moveit packages
    from moveit_msgs.srv import *
    from moveit_msgs.msg import *

    from mujoco_moveit_connector.srv import *
except ModuleNotFoundError:
    print("ROS dependency errors. Robot path planning will be unable.")

# from robosuite.mcts.transforms import *
from robosuite.utils import transform_utils as tf

from matplotlib import pyplot as plt


def get_meshes(_mesh_types=None, _mesh_files=None, _mesh_units=None, _area_ths=0.003, _tbl_area_ths=0.003, _rotation_types=4):

    if _mesh_types is None:
        _mesh_types = ['arch_box',
                       'rect_box',
                       'square_box',
                       'half_cylinder_box',
                       'triangle_box',
                       'twin_tower_goal',
                       'tower_goal',
                       'box_goal',
                       'custom_table']
    if _mesh_files is None:
        _mesh_files = ['./robosuite/models/assets/objects/meshes/arch_box.stl',
                       './robosuite/models/assets/objects/meshes/rect_box.stl',
                       './robosuite/models/assets/objects/meshes/square_box.stl',
                       './robosuite/models/assets/objects/meshes/half_cylinder_box.stl',
                       './robosuite/models/assets/objects/meshes/triangle_box.stl',
                       './robosuite/models/assets/objects/meshes/twin_tower_goal.stl',
                       './robosuite/models/assets/objects/meshes/tower_goal.stl',
                       './robosuite/models/assets/objects/meshes/box_goal.stl',
                       './robosuite/models/assets/objects/meshes/custom_table.stl']
    if _mesh_units is None:
        _mesh_units = [0.001, 0.001, 0.001, 0.001, 0.001, 0.0011, 0.001, 0.0011, 0.01]

    _meshes = []
    _contact_points = []
    _contact_faces = []

    for mesh_type, mesh_file, unit in zip(_mesh_types, _mesh_files, _mesh_units):
        mesh = trimesh.load(mesh_file)

        mesh.apply_scale(unit)
        mesh.apply_translation(-mesh.center_mass)
        while True:
            indices, = np.where(mesh.area_faces > _area_ths)
            if len(indices) > 0:
                mesh = mesh.subdivide(indices)
            else:
                break

        if 'table' in mesh_type:
            while True:
                indices, = np.where(mesh.area_faces > _tbl_area_ths)
                if len(indices) > 0:
                    mesh = mesh.subdivide(indices)
                else:
                    break

            points = np.mean(mesh.vertices[mesh.faces], axis=1)
            # find top surfaces of the mesh
            faces, = np.where(points[:, 2] == np.max(points, axis=0)[2])
            points = points[faces]
        elif 'goal' in mesh_type:
            points = np.mean(mesh.vertices[mesh.faces], axis=1)
            # find bottom surfaces of the mesh
            faces, = np.where(
                np.logical_and(mesh.face_normals[:, 2] < - 0.99, points[:, 2] == np.min(points, axis=0)[2]))
            points = points[faces]
        else:
            faces = np.arange(mesh.faces.shape[0])
            points = np.mean(mesh.vertices[mesh.faces], axis=1)

        _meshes.append(mesh)
        _contact_faces.append(faces)
        _contact_points.append(points)

    return _mesh_types, _mesh_files, _mesh_units, _meshes, _rotation_types, _contact_faces, _contact_points


class Object(object):
    def __init__(self, _name, _mesh_idx, _pose, _logical_state):
        self.name = _name
        self.mesh_idx = _mesh_idx
        self.pose = _pose
        self.logical_state = _logical_state
        self.color = [np.random.uniform(), np.random.uniform(), np.random.uniform(), 0.3]


def get_obj_idx_by_name(_object_list, _name):
    for _obj_idx, _obj in enumerate(_object_list):
        if _obj.name == _name:
            return _obj_idx
    return None


def get_held_object(_object_list):
    for _obj_idx, _obj in enumerate(_object_list):
        if "held" in _obj.logical_state: return _obj_idx
    return None


def update_logical_state(_object_list):
    for _obj in _object_list:
        if "on" in _obj.logical_state:
            for _support_obj_name in _obj.logical_state["on"]:
                _support_obj_idx = get_obj_idx_by_name(_object_list, _support_obj_name)
                if "support" in _object_list[_support_obj_idx].logical_state:
                    if _obj.name in _object_list[_support_obj_idx].logical_state["support"]:
                        continue
                    else:
                        _object_list[_support_obj_idx].logical_state["support"].append(_obj.name)
                else:
                    _object_list[_support_obj_idx].logical_state["support"] = [_obj.name]

        if "support" in _obj.logical_state:
            for _on_obj_name in _obj.logical_state["support"]:
                _on_obj_idx = get_obj_idx_by_name(_object_list, _on_obj_name)
                if "on" in _object_list[_on_obj_idx].logical_state:
                    if _obj.name in _object_list[_on_obj_idx].logical_state["on"]:
                        continue
                    else:
                        _object_list[_on_obj_idx].logical_state["on"].append(_obj.name)
                else:
                    _object_list[_on_obj_idx].logical_state["on"] = [_obj.name]


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def rotation_matrix_from_z_x(z_axis):
    x_axis = [1, 0, 0]
    y_axis = np.cross(z_axis, x_axis)
    if np.sqrt(np.sum(y_axis ** 2)) > 0.:
        y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))
    else:
        x_axis = [0, 1, 0]
        y_axis = np.cross(z_axis, x_axis)
        if np.sqrt(np.sum(y_axis ** 2)) > 0.:
            y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))
        else:
            x_axis = [0, 0, 1]
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis

    return T


def rotation_matrix_from_y_x(y_axis):
    x_axis = [1, 0, 0]
    z_axis = np.cross(x_axis, y_axis)
    if np.sqrt(np.sum(z_axis ** 2)) > 0.:
        z_axis = z_axis / np.sqrt(np.sum(z_axis ** 2))
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))
    else:
        x_axis = [0, 1, 0]
        z_axis = np.cross(x_axis, y_axis)
        if np.sqrt(np.sum(z_axis ** 2)) > 0.:
            z_axis = z_axis / np.sqrt(np.sum(z_axis ** 2))
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))
        else:
            x_axis = [0, 0, 1]
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.sqrt(np.sum(z_axis ** 2))
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))

    rot_mtx = np.eye(4)
    rot_mtx[:3, 0] = x_axis
    rot_mtx[:3, 1] = y_axis
    rot_mtx[:3, 2] = z_axis

    return rot_mtx


def transform_matrix2pose(pose_mtx):
    q = tf.mat2quat(pose_mtx)
    orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    position = Point(x=pose_mtx[0, 3], y=pose_mtx[1, 3], z=pose_mtx[2, 3])
    pose = Pose(position, orientation)

    return pose


def pose2transform_matrix(pose):
    orientation = pose.orientation
    position = pose.position

    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    pose_mtx = np.eye(4)
    pose_mtx[:3, :3] = tf.quat2mat(q)
    pose_mtx[0, 3] = position.x
    pose_mtx[1, 3] = position.y
    pose_mtx[2, 3] = position.z

    return pose_mtx


def get_grasp_pose(_mesh_idx1, _pnt1, _normal1, _rotation1, _pose1, _meshes):
    mesh1 = _meshes[_mesh_idx1]
    locations, index_ray, index_tri = mesh1.ray.intersects_location(
        ray_origins=[_pnt1 - 1e-5 * _normal1],
        ray_directions=[-_normal1])

    _pnt2 = locations[0]
    _pnt_center = (_pnt1 + _pnt2) / 2.

    y_axis = _pnt2 - _pnt1
    gripper_width = np.sqrt(np.sum(y_axis ** 2))
    y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))

    t_grasp = rotation_matrix_from_y_x(y_axis).dot(rotation_matrix([0, 1, 0], _rotation1))
    t_grasp[:3, 3] = _pnt_center

    t_retreat = deepcopy(t_grasp)
    approaching_dir = t_retreat[:3, 2]
    t_retreat[:3, 3] = _pnt_center - 3e-2 * approaching_dir
    hand_t_grasp = _pose1.dot(t_grasp)
    hand_t_retreat = _pose1.dot(t_retreat)
    # if hand_t_grasp[2, 2] < 1e-10:
    if hand_t_grasp[2, 2] < -0.99:
        return hand_t_grasp, hand_t_retreat, gripper_width
    else:
        return None, None, None


def get_on_pose(_name1, _pnt1, _normal1, _rotation1, _pnt2, _normal2, _pose2, _coll_mngr):
    target_pnt = _pnt1 + 1e-6 * _normal1
    T_target = rotation_matrix_from_z_x(-_normal1).dot(rotation_matrix([0, 0, 1], _rotation1))
    T_target[:3, 3] = target_pnt

    T_source = rotation_matrix_from_z_x(_normal2)
    T_source[:3, 3] = _pnt2

    _T1 = _pose2.dot(T_source.dot(np.linalg.inv(T_target)))
    _coll_mngr.set_transform(_name1, _T1)
    if not _coll_mngr.in_collision_internal():
        return _T1
    else:
        return None


def configuration_initializer(_mesh_types, _meshes, _mesh_units, _rotation_types, _contact_faces, _contact_points,
                              goal_name='tower_goal'):
    _object_list = []
    _coll_mngr = trimesh.collision.CollisionManager()

    if 'tower_goal' is goal_name:
        n_obj_per_mesh_types = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    elif 'twin_tower_goal' is goal_name:
        n_obj_per_mesh_types = [0, 2, 2, 0, 2, 0, 0, 0, 0, 0]
    elif 'box_goal' is goal_name:
        n_obj_per_mesh_types = [0, 2, 4, 0, 0, 0, 0, 0, 0, 0]
    elif 'stack_easy' is goal_name:
        n_obj_per_mesh_types = [0, 2, 2, 0, 0, 0, 0, 0, 0, 0]
    elif 'stack_hard' is goal_name:
        n_obj_per_mesh_types = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    elif 'regular_shapes' is goal_name:
        n_obj_per_mesh_types = [0, 2, 2, 0, 0, 0, 0, 0, 0, 0]
    elif 'round_shapes' is goal_name:
        n_obj_per_mesh_types = [2, 0, 0, 2, 2, 0, 0, 0, 0, 0]
    else:
        assert Exception('goal name is wrong!!!')

    if 'goal' in goal_name:
        table_spawn_position = [0.4, 0.25]
        table_spawn_bnd_size = 0.1
    else:
        table_spawn_position = [0.4, 0.0]
        table_spawn_bnd_size = 0.15

    table_name = 'custom_table'
    table_pose = np.array([0.6, 0.0, 0.573077])
    table_mesh_idx = _mesh_types.index(table_name)
    table_T = np.eye(4)
    table_T[:3, 3] = table_pose
    table_obj = Object(table_name, table_mesh_idx, table_T, {"static": []})

    _object_list.append(table_obj)
    _coll_mngr.add_object(table_name, _meshes[table_mesh_idx])
    _coll_mngr.set_transform(table_name, table_T)

    table_contact_points = _contact_points[table_mesh_idx]
    table_contact_normals = _meshes[table_mesh_idx].face_normals[_contact_faces[table_mesh_idx]]

    table_contact_points_world = table_T[:3, :3].dot(table_contact_points.T).T + table_T[:3, 3]
    table_contact_normals_world = table_T[:3, :3].dot(table_contact_normals.T).T
    table_contact_indices, = np.where(
        np.logical_and(
            np.logical_and(
                table_contact_normals_world[:, 2] > 0.,
                np.abs(table_spawn_position[0] - table_contact_points_world[:, 0]) < table_spawn_bnd_size
            ), np.abs(table_spawn_position[1] - table_contact_points_world[:, 1]) < table_spawn_bnd_size
        )
    )

    _contact_points[table_mesh_idx] = _contact_points[table_mesh_idx][table_contact_indices]
    _contact_faces[table_mesh_idx] = _contact_faces[table_mesh_idx][table_contact_indices]

    if 'goal' in goal_name:
        goal_mesh_idx = _mesh_types.index(goal_name)
        goal_pose = np.array([0.4, 0.0, -0.019*_mesh_units[goal_mesh_idx]*1000 + _meshes[table_mesh_idx].bounds[1][2] - _meshes[table_mesh_idx].bounds[0][2]
                              - _meshes[goal_mesh_idx].bounds[0][2]])
        goal_T = np.eye(4)
        goal_T[:3, 3] = goal_pose
        _goal_obj = Object(goal_name, goal_mesh_idx, goal_T, {"goal": []})
    else:
        _goal_obj = None

    for mesh_idx, n_obj in enumerate(n_obj_per_mesh_types):
        obj_mesh_idx = mesh_idx
        for obj_idx in range(n_obj):
            obj_name = _mesh_types[mesh_idx] + str(obj_idx)

            # Find stable pose
            if "custom_table" not in _mesh_types[mesh_idx]:
                stable_poses, probs = _meshes[mesh_idx].compute_stable_poses(n_samples=100)
                stable_pose_idx = np.argmax(probs)
                stable_pose = stable_poses[stable_pose_idx]

                if "half_cylinder_box" in _mesh_types[mesh_idx] or "triangle_box" in _mesh_types[mesh_idx]:
                    stable_pose = rotation_matrix([1., 0., 0.], -np.pi / 2.).dot(stable_pose)
                elif "arch_box" in _mesh_types[mesh_idx]:
                    stable_pose = rotation_matrix([1., 0., 0.], np.pi / 2.).dot(stable_pose)

                stable_pose = stable_pose.dot(rotation_matrix([0., 1., 0.], np.pi / 2.))
            else:
                stable_pose = np.eye(4)

            new_obj = Object(obj_name, obj_mesh_idx, stable_pose, {"on": [table_name]})
            _coll_mngr.add_object(obj_name, _meshes[obj_mesh_idx])

            new_obj_contact_points = _contact_points[obj_mesh_idx]
            new_obj_contact_normals = _meshes[obj_mesh_idx].face_normals[_contact_faces[obj_mesh_idx]]

            new_obj_contact_normals_world = stable_pose[:3, :3].dot(new_obj_contact_normals.T).T
            new_obj_contact_indices, = np.where(new_obj_contact_normals_world[:, 2] < -0.99)

            table_contact_points = _contact_points[table_mesh_idx]
            table_contact_normals = _meshes[table_mesh_idx].face_normals[_contact_faces[table_mesh_idx]]

            table_contact_normals_world = table_T[:3, :3].dot(table_contact_normals.T).T
            table_contact_indices, = np.where(table_contact_normals_world[:, 2] > 0.)

            pose1_list = []
            for i in new_obj_contact_indices:
                for j in range(_rotation_types):
                    for k in table_contact_indices:
                        pnt1 = new_obj_contact_points[i]
                        normal1 = new_obj_contact_normals[i]

                        pnt2 = table_contact_points[k]
                        normal2 = table_contact_normals[k]

                        pose1 = get_on_pose(new_obj.name, pnt1, normal1, 2.*np.pi*j/_rotation_types, pnt2, normal2,
                                            table_T, _coll_mngr)
                        if pose1 is not None:
                            min_dist1 = np.min([np.sqrt(np.sum(np.square(tmp_obj.pose[:3, 3] - pose1[:3, 3]))) for tmp_obj in _object_list if tmp_obj.name not in new_obj.name])
                            if min_dist1 > 0.05:
                                pose1_list.append(pose1)
            print(new_obj.name, len(pose1_list))
            if len(pose1_list) > 0:
                init_pose_idx = np.random.choice(len(pose1_list), 1)[0]
                new_obj.pose = pose1_list[init_pose_idx]
                _coll_mngr.set_transform(new_obj.name, new_obj.pose)
            _object_list.append(new_obj)
            update_logical_state(_object_list)

    if _goal_obj is not None:
        transform_g_t = np.linalg.inv(table_obj.pose).dot(_goal_obj.pose)
        _contact_points[table_mesh_idx] = transform_g_t[:3, :3].dot(_contact_points[goal_mesh_idx].T).T \
                                         + transform_g_t[:3, 3]
        _contact_faces[table_mesh_idx] = _contact_faces[table_mesh_idx][:len(_contact_faces[goal_mesh_idx])]
        # this is the most tricky part...
    return _object_list, _goal_obj, _contact_points, _contact_faces, _coll_mngr, table_spawn_position, \
           table_spawn_bnd_size, n_obj_per_mesh_types


def visualize(_object_list, _meshes, _goal_obj=None):
    mesh_scene = trimesh.Scene()
    for _obj in _object_list:
        mesh_scene.add_geometry(_meshes[_obj.mesh_idx], node_name=_obj.name, transform=_obj.pose)
    if _goal_obj is not None:
        mesh_scene.add_geometry(_meshes[_goal_obj.mesh_idx], node_name=_goal_obj.name, transform=_goal_obj.pose)
    mesh_scene.show()


def add_supporting_object(obj_idx, object_list, meshes, trimesh_scene):
    trimesh_scene.add_geometry(meshes[object_list[obj_idx].mesh_idx], node_name=object_list[obj_idx].name,
                               transform=object_list[obj_idx].pose)
    if "on" in object_list[obj_idx].logical_state:
        for obj_name in object_list[obj_idx].logical_state["on"]:
            support_obj_idx = get_obj_idx_by_name(object_list, obj_name)
            trimesh_scene = add_supporting_object(support_obj_idx, object_list, meshes, trimesh_scene)
    return trimesh_scene


def get_image(object_list, action, next_object_list, meshes, label=None, do_visualize=False):

    place_idx = get_obj_idx_by_name(next_object_list, action['param'])
    pick_idx = get_held_object(object_list)

    place_translation = deepcopy(next_object_list[pick_idx].pose[:3, 3])
    place_translation[2] -= 0.06

    trimesh_scene = trimesh.Scene()
    trimesh_scene = add_supporting_object(place_idx, next_object_list, meshes, trimesh_scene)
    trimesh_scene.add_geometry(meshes[next_object_list[pick_idx].mesh_idx], node_name=next_object_list[pick_idx].name,
                               transform=next_object_list[pick_idx].pose)

    scene = pyrender.Scene.from_trimesh_scene(trimesh_scene)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose1 = np.eye(4)
    camera_pose2 = np.eye(4)

    camera_pose1[:3, :3] = tf.euler2mat([1.9 * np.pi / 4., 0., 0.])
    camera_pose1[:3, 3] = place_translation
    camera_pose1[1, 3] -= 0.2
    camera_pose1[2, 3] += 0.05

    camera_pose2[:3, :3] = tf.euler2mat([0., 1.9 * np.pi / 4., np.pi / 2.])
    camera_pose2[:3, 3] = place_translation
    camera_pose2[0, 3] += 0.2
    camera_pose2[2, 3] += 0.05

    scene.add(camera, pose=camera_pose1, name='cam1')
    scene.add(camera, pose=camera_pose2, name='cam2')

    light = pyrender.SpotLight(color=np.ones(3), intensity=1.,
                               innerConeAngle=np.pi / 8.0,
                               outerConeAngle=np.pi / 2.0)
    for obj in object_list:
        if "custom_table" in obj.name:
            table_pose1 = deepcopy(obj.pose)
            table_pose2 = deepcopy(obj.pose)
            table_pose1[0, 3] += .5
            table_pose1[2, 3] += 1.
            table_pose2[0, 3] -= .5
            table_pose2[2, 3] += 1.

    scene.add(light, pose=table_pose1)
    scene.add(light, pose=table_pose2)

    colors = []
    depths = []
    masks = []
    for camera_name in ['cam1', 'cam2']:
        for camera_node in scene.camera_nodes:
            if camera_node.name == camera_name:
                scene.main_camera_node = camera_node
        r = pyrender.OffscreenRenderer(128, 128)
        color, depth = r.render(scene)

        for i, node in enumerate(scene.mesh_nodes):
            if np.sum(np.abs(next_object_list[pick_idx].pose - node.matrix)) < 1e-7:
                node.mesh.is_visible = True
            else:
                node.mesh.is_visible = False

        _depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        mask = np.logical_and(
            (np.abs(_depth - depth) < 1e-6), np.abs(depth) > 0
        )

        for mn in scene.mesh_nodes:
            mn.mesh.is_visible = True

        colors.append(color)
        depths.append(depth)
        masks.append(mask)

    if do_visualize:
        plt.figure(figsize=(72, 36))
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.imshow(colors[0])
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.imshow(depths[0])
        plt.subplot(2, 3, 3)
        plt.axis('off')
        plt.imshow(masks[0])
        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.imshow(colors[1])
        plt.subplot(2, 3, 5)
        plt.axis('off')
        plt.imshow(depths[1])
        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.imshow(masks[1])
        plt.title(label)
        plt.show()

    return colors, depths, masks


def get_possible_actions(_object_list, _meshes, _coll_mngr, _contact_points, _contact_faces, _rotation_types, side_place_flag=False):
    _action_list = []
    for obj in _object_list:
        _coll_mngr.set_transform(obj.name, obj.pose)

    # Check pick
    if all(["held" not in obj.logical_state for obj in _object_list]):
        for obj1 in _object_list:
            if "support" not in obj1.logical_state and "static" not in obj1.logical_state and \
                    "done" not in obj1.logical_state and "goal" not in obj1.name:
                _action_list.append({"type": "pick", "param": obj1.name})

    # Check place
    held_obj_idx = get_held_object(_object_list)
    if held_obj_idx is not None:
        held_obj = _object_list[held_obj_idx]
        obj2 = held_obj
        obj2_mesh_idx = held_obj.mesh_idx
        obj2_contact_points = _contact_points[obj2_mesh_idx]
        obj2_contact_normals = _meshes[obj2_mesh_idx].face_normals[_contact_faces[obj2_mesh_idx]]

        obj2_contact_normals_world = obj2.pose[:3, :3].dot(obj2_contact_normals.T).T
        if side_place_flag:
            obj2_contact_indices, = np.where(obj2_contact_normals_world[:, 2] < 1e-3)
        else:
            obj2_contact_indices, = np.where(obj2_contact_normals_world[:, 2] < 0.)

        for obj1 in _object_list:
            if "held" not in obj1.logical_state:
                obj1_mesh_idx = obj1.mesh_idx
                obj1_contact_points = _contact_points[obj1_mesh_idx]
                obj1_contact_normals = _meshes[obj1_mesh_idx].face_normals[_contact_faces[obj1_mesh_idx]]

                obj1_contact_normals_world = obj1.pose[:3, :3].dot(obj1_contact_normals.T).T
                obj1_contact_indices, = np.where(obj1_contact_normals_world[:, 2] > 0.99)

                for i in obj1_contact_indices:
                    for j in range(_rotation_types):
                        for k in obj2_contact_indices:
                            pnt1 = obj1_contact_points[i]
                            normal1 = obj1_contact_normals[i]

                            pnt2 = obj2_contact_points[k]
                            normal2 = obj2_contact_normals[k]

                            pose = get_on_pose(obj2.name, pnt2, normal2, 2. * np.pi * j / _rotation_types,
                                               pnt1, normal1, obj1.pose, _coll_mngr)

                            if pose is not None:
                                _action_list.append({"type": "place", "param": obj1.name, "placing_pose": pose})
    return _action_list


def get_possible_actions_with_robot(_object_list, _meshes, _coll_mngr, _contact_points, _contact_faces, _rotation_types, side_place_flag=False):
    _action_list = []
    for obj in _object_list:
        _coll_mngr.set_transform(obj.name, obj.pose)

    # Check pick
    if all(["held" not in obj.logical_state for obj in _object_list]):
        for obj1 in _object_list:
            if "support" not in obj1.logical_state and "static" not in obj1.logical_state and \
                    "done" not in obj1.logical_state and "goal" not in obj1.name:
                obj1_mesh_idx = obj1.mesh_idx
                obj1_pose = obj1.pose
                obj1_contact_points = _contact_points[obj1_mesh_idx]
                obj1_contact_normals = _meshes[obj1_mesh_idx].face_normals[_contact_faces[obj1_mesh_idx]]

                obj1_contact_normals_world = obj1.pose[:3, :3].dot(obj1_contact_normals.T).T
                obj1_contact_indices, = np.where(np.abs(obj1_contact_normals_world[:, 2]) < 1e-10)

                for i in obj1_contact_indices:
                    for j in range(_rotation_types):
                        pnt1 = obj1_contact_points[i]
                        normal1 = obj1_contact_normals[i]
                        hand_t_grasp, hand_t_retreat, gripper_width = get_grasp_pose(obj1_mesh_idx, pnt1, normal1, 2. * np.pi * j / _rotation_types, obj1_pose, _meshes)

                        if hand_t_grasp is not None:
                            _action_list.append({"type": "pick", "param": obj1.name, "grasp_pose": hand_t_grasp,
                                                 "retreat_pose": hand_t_retreat, "gripper_width": gripper_width})

    # Check place
    held_obj_idx = get_held_object(_object_list)
    if held_obj_idx is not None:
        held_obj = _object_list[held_obj_idx]
        obj2 = held_obj
        obj2_mesh_idx = held_obj.mesh_idx
        obj2_contact_points = _contact_points[obj2_mesh_idx]
        obj2_contact_normals = _meshes[obj2_mesh_idx].face_normals[_contact_faces[obj2_mesh_idx]]

        obj2_contact_normals_world = obj2.pose[:3, :3].dot(obj2_contact_normals.T).T
        if side_place_flag:
            obj2_contact_indices, = np.where(obj2_contact_normals_world[:, 2] < 1e-3)
        else:
            obj2_contact_indices, = np.where(obj2_contact_normals_world[:, 2] < 0.)

        for obj1 in _object_list:
            if "held" not in obj1.logical_state:
                obj1_mesh_idx = obj1.mesh_idx
                obj1_contact_points = _contact_points[obj1_mesh_idx]
                obj1_contact_normals = _meshes[obj1_mesh_idx].face_normals[_contact_faces[obj1_mesh_idx]]

                obj1_contact_normals_world = obj1.pose[:3, :3].dot(obj1_contact_normals.T).T
                obj1_contact_indices, = np.where(obj1_contact_normals_world[:, 2] > 0.99)

                for i in obj1_contact_indices:
                    for j in range(_rotation_types):
                        for k in obj2_contact_indices:
                            pnt1 = obj1_contact_points[i]
                            normal1 = obj1_contact_normals[i]

                            pnt2 = obj2_contact_points[k]
                            normal2 = obj2_contact_normals[k]

                            pose = get_on_pose(obj2.name, pnt2, normal2, 2. * np.pi * j / _rotation_types,
                                               pnt1, normal1, obj1.pose, _coll_mngr)

                            if pose is not None:
                                _action_list.append({"type": "place", "param": obj1.name, "placing_pose": pose})
    return _action_list


def get_possible_actions_v2(_object_list, _meshes, _coll_mngr, _contact_points, _contact_faces, _rotation_types, side_place_flag=False):
    _action_list = []
    for obj in _object_list:
        _coll_mngr.set_transform(obj.name, obj.pose)

    # Check pick
    if all(["held" not in obj.logical_state for obj in _object_list]):
        for obj1 in _object_list:
            if "support" not in obj1.logical_state and "static" not in obj1.logical_state and \
                    "done" not in obj1.logical_state and "goal" not in obj1.name:
                obj1_mesh_idx = obj1.mesh_idx
                obj1_pose = obj1.pose
                obj1_contact_points = _contact_points[obj1_mesh_idx]
                obj1_contact_normals = _meshes[obj1_mesh_idx].face_normals[_contact_faces[obj1_mesh_idx]]

                obj1_contact_normals_world = obj1.pose[:3, :3].dot(obj1_contact_normals.T).T
                obj1_contact_indices, = np.where(np.abs(obj1_contact_normals_world[:, 2]) < 1e-10)

                grasp_poses = []
                retreat_poses = []
                gripper_widths = []
                for i in obj1_contact_indices:
                    for j in range(_rotation_types):
                        pnt1 = obj1_contact_points[i]
                        normal1 = obj1_contact_normals[i]
                        hand_t_grasp, hand_t_retreat, gripper_width = get_grasp_pose(obj1_mesh_idx, pnt1, normal1, 2. * np.pi * j / _rotation_types, obj1_pose, _meshes)

                        if hand_t_grasp is not None:
                            grasp_poses.append(hand_t_grasp)
                            retreat_poses.append(hand_t_retreat)
                            gripper_widths.append(gripper_width)
                _action_list.append({"type": "pick", "param": obj1.name, "grasp_poses": grasp_poses,
                                     "retreat_poses": retreat_poses, "gripper_widths": gripper_widths})

    # Check place
    held_obj_idx = get_held_object(_object_list)
    if held_obj_idx is not None:
        held_obj = _object_list[held_obj_idx]
        obj2 = held_obj
        obj2_mesh_idx = held_obj.mesh_idx
        obj2_contact_points = _contact_points[obj2_mesh_idx]
        obj2_contact_normals = _meshes[obj2_mesh_idx].face_normals[_contact_faces[obj2_mesh_idx]]

        obj2_contact_normals_world = obj2.pose[:3, :3].dot(obj2_contact_normals.T).T
        if side_place_flag:
            obj2_contact_indices, = np.where(obj2_contact_normals_world[:, 2] < 1e-3)
        else:
            obj2_contact_indices, = np.where(obj2_contact_normals_world[:, 2] < 0.)

        for obj1 in _object_list:
            if "held" not in obj1.logical_state:
                obj1_mesh_idx = obj1.mesh_idx
                obj1_contact_points = _contact_points[obj1_mesh_idx]
                obj1_contact_normals = _meshes[obj1_mesh_idx].face_normals[_contact_faces[obj1_mesh_idx]]

                obj1_contact_normals_world = obj1.pose[:3, :3].dot(obj1_contact_normals.T).T
                obj1_contact_indices, = np.where(obj1_contact_normals_world[:, 2] > 0.99)

                for i in obj1_contact_indices:
                    for j in range(_rotation_types):
                        for k in obj2_contact_indices:
                            pnt1 = obj1_contact_points[i]
                            normal1 = obj1_contact_normals[i]

                            pnt2 = obj2_contact_points[k]
                            normal2 = obj2_contact_normals[k]

                            pose = get_on_pose(obj2.name, pnt2, normal2, 2. * np.pi * j / _rotation_types,
                                               pnt1, normal1, obj1.pose, _coll_mngr)

                            if pose is not None:
                                _action_list.append({"type": "place", "param": obj1.name, "placing_pose": pose})
    return _action_list


def get_transition(_object_list, _action):
    next_object_list = deepcopy(_object_list)
    if _action["type"] is "pick":
        pick_obj_idx = get_obj_idx_by_name(next_object_list, _action['param'])
        next_object_list[pick_obj_idx].logical_state.clear()
        next_object_list[pick_obj_idx].logical_state["held"] = []
        update_logical_state(next_object_list)

    elif _action["type"] is "place":
        place_obj_idx = get_obj_idx_by_name(next_object_list, _action['param'])
        held_obj_idx = get_held_object(next_object_list)
        support_obj_idx = get_obj_idx_by_name(next_object_list, next_object_list[held_obj_idx].logical_state["on"][0])
        new_pose = _action["placing_pose"]
        if new_pose is None:
            return None

        next_object_list[held_obj_idx].pose = new_pose
        next_object_list[support_obj_idx].logical_state["support"].remove(next_object_list[held_obj_idx].name)
        next_object_list[held_obj_idx].logical_state.clear()
        next_object_list[held_obj_idx].logical_state["on"] = [next_object_list[place_obj_idx].name]
        next_object_list[held_obj_idx].logical_state["done"] = []
        update_logical_state(next_object_list)

    return next_object_list


def get_reward(_obj_list, _action, _goal_obj, _next_obj_list, _meshes):
    if _next_obj_list is None:
        return -np.inf
    elif "pick" in _action["type"]:
        return 0.0
    elif "place" in _action["type"]:
        if _goal_obj is None:
            obj_height_list = []
            for obj in _obj_list:
                obj_height_list.append(obj.pose[2, 3])
            curr_height = np.max(obj_height_list)

            next_obj_height_list = []
            for next_obj in _next_obj_list:
                next_obj_height_list.append(next_obj.pose[2, 3])
            next_height = np.max(next_obj_height_list)
            return next_height - curr_height
        else:
            goal_mesh_copied = deepcopy(_meshes[_goal_obj.mesh_idx])
            goal_mesh_copied.apply_transform(_goal_obj.pose)

            rew = 0.
            for obj in _obj_list:
                if "table" not in obj.name:
                    obj_mesh_copied = deepcopy(_meshes[obj.mesh_idx])
                    obj_mesh_copied.apply_transform(obj.pose)

                    signed_distance = trimesh.proximity.signed_distance(goal_mesh_copied, obj_mesh_copied.vertices)
                    rew -= 1 / (1 + 100 * np.max(np.abs(signed_distance)))

            for obj in _next_obj_list:
                if "table" not in obj.name:
                    obj_mesh_copied = deepcopy(_meshes[obj.mesh_idx])
                    obj_mesh_copied.apply_transform(obj.pose)

                    signed_distance = trimesh.proximity.signed_distance(goal_mesh_copied, obj_mesh_copied.vertices)
                    rew += 1 / (1 + 100 * np.max(np.abs(signed_distance)))

            return rew


def synchronize_planning_scene(_left_joint_values, _left_gripper_width, _right_joint_values, _right_gripper_width,
                               _object_list, _meshes, _get_planning_scene_proxy=None, _apply_planning_scene_proxy=None):
    joint_values = {}
    joint_values.update(_left_joint_values)
    joint_values.update(_right_joint_values)

    resp = _get_planning_scene_proxy(GetPlanningSceneRequest())
    current_scene = resp.scene

    next_scene = deepcopy(current_scene)
    next_scene.robot_state.joint_state.name = joint_values.keys()
    next_scene.robot_state.joint_state.position = joint_values.values()
    next_scene.robot_state.joint_state.velocity = [0] * len(joint_values)
    next_scene.robot_state.joint_state.effort = [0] * len(joint_values)
    next_scene.robot_state.is_diff = True

    held_obj_idx = get_held_object(_object_list)
    if held_obj_idx is None and len(next_scene.robot_state.attached_collision_objects) > 0:
        for attached_object_idx in range(len(next_scene.robot_state.attached_collision_objects)):
            next_scene.robot_state.attached_collision_objects[
                attached_object_idx].object.operation = CollisionObject.REMOVE

    for obj in _object_list:
        co_idx = None
        for scene_object_idx in range(len(next_scene.world.collision_objects)):
            if next_scene.world.collision_objects[scene_object_idx].id in obj.name:
                co_idx = scene_object_idx
                break

        if co_idx is None:
            co = CollisionObject()
            co.operation = CollisionObject.ADD
            co.id = obj.name
            co.header = next_scene.robot_state.joint_state.header

            mesh = Mesh()
            for face in _meshes[obj.mesh_idx].faces:
                triangle = MeshTriangle()
                triangle.vertex_indices = face
                mesh.triangles.append(triangle)

            for vertex in _meshes[obj.mesh_idx].vertices:
                point = Point()
                point.x = vertex[0]
                point.y = vertex[1]
                point.z = vertex[2]
                mesh.vertices.append(point)

            co.meshes = [mesh]
            obj_pose = transform_matrix2pose(obj.pose)
            obj_pose.position.z -= 0.93
            co.mesh_poses = [obj_pose]

            if held_obj_idx is not None and len(next_scene.robot_state.attached_collision_objects) == 0 and \
                    _object_list[held_obj_idx].name is obj.name:
                aco = AttachedCollisionObject()
                aco.link_name = 'left_gripper'
                aco.object = co
                next_scene.robot_state.attached_collision_objects.append(aco)
            elif held_obj_idx is not None and len(next_scene.robot_state.attached_collision_objects) > 0 and \
                    _object_list[held_obj_idx].name is obj.name:
                continue
            else:
                next_scene.world.collision_objects.append(co)

        else:
            next_scene.world.collision_objects[co_idx].operation = CollisionObject.MOVE
            next_scene.world.collision_objects[co_idx].header = next_scene.robot_state.joint_state.header
            obj_pose = transform_matrix2pose(obj.pose)
            obj_pose.position.z -= 0.93
            next_scene.world.collision_objects[co_idx].mesh_poses = [obj_pose]

            if held_obj_idx is not None and len(next_scene.robot_state.attached_collision_objects) == 0 and \
                    _object_list[held_obj_idx].name in obj.name:
                aco = AttachedCollisionObject()
                aco.link_name = 'left_gripper'
                aco.object = deepcopy(next_scene.world.collision_objects[co_idx])
                aco.object.operation = CollisionObject.ADD
                next_scene.robot_state.attached_collision_objects.append(aco)
                next_scene.world.collision_objects[co_idx].operation = CollisionObject.REMOVE

    next_scene.robot_state.joint_state.name = list(next_scene.robot_state.joint_state.name) \
                                              + ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint']
    next_scene.robot_state.joint_state.position = list(next_scene.robot_state.joint_state.position) \
                                                  + [np.minimum(0.03, _left_gripper_width),
                                                     -np.minimum(0.03, _left_gripper_width)]

    next_scene.robot_state.joint_state.name = list(next_scene.robot_state.joint_state.name) \
                                              + ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint']
    next_scene.robot_state.joint_state.position = list(next_scene.robot_state.joint_state.position)  \
                                                  + [np.minimum(0.03, _right_gripper_width),
                                                     -np.minimum(0.03, _right_gripper_width)]

    # print("------------World Objects")
    # for scene_object_idx in range(len(next_scene.world.collision_objects)):
    #     print(next_scene.world.collision_objects[scene_object_idx].id, next_scene.world.collision_objects[scene_object_idx].operation)
    # print("------------Attached Objects")
    # for attached_object_idx in range(len(next_scene.robot_state.attached_collision_objects)):
    #     print(next_scene.robot_state.attached_collision_objects[attached_object_idx].object.id, next_scene.robot_state.attached_collision_objects[attached_object_idx].object.operation)
    # print("=================================")

    next_scene.is_diff = True
    req = ApplyPlanningSceneRequest()
    req.scene = next_scene
    resp = _apply_planning_scene_proxy(req)
    for _ in range(100):
        rospy.sleep(0.001)


def kinematic_planning(_object_list,
                       _left_joint_values, _left_gripper_width,
                       _right_joint_values, _right_gripper_width,
                       _action, _meshes,
                       _get_planning_scene_proxy=None,
                       _apply_planning_scene_proxy=None,
                       _planning_with_gripper_pose_proxy=None,
                       _planning_with_arm_joints_proxy=None,
                       _compute_fk_proxy=None,
                       _init_left_joint_values=None,
                       _init_left_gripper_width=0.03,
                       _init_right_joint_values=None,
                       _init_right_gripper_width=0.03):

    if _init_right_joint_values is None:
        _init_right_joint_values = {'right_w0': -0.6699952259595108,
                                    'right_w1': 1.030009435085784,
                                    'right_w2': 0.4999997247485215,
                                    'right_e0': 1.189968899785275,
                                    'right_e1': 1.9400238130755056,
                                    'right_s0': -0.08000397926829805,
                                    'right_s1': -0.9999781166910306}
    if _init_left_joint_values is None:
        _init_left_joint_values = {'left_w0': 0.6699952259595108,
                                   'left_w1': 1.030009435085784,
                                   'left_w2': -0.4999997247485215,
                                   'left_e0': -1.189968899785275,
                                   'left_e1': 1.9400238130755056,
                                   'left_s0': -0.08000397926829805,
                                   'left_s1': -0.9999781166910306}

    synchronize_planning_scene(_left_joint_values, _left_gripper_width, _right_joint_values, _right_gripper_width,
                               _object_list, _meshes,
                               _get_planning_scene_proxy=_get_planning_scene_proxy,
                               _apply_planning_scene_proxy=_apply_planning_scene_proxy)

    _next_object_list = deepcopy(_object_list)
    planned_traj_list = []
    if _action["type"] is "pick":

        grasp_poses = _action["grasp_poses"]
        gripper_widths = _action["gripper_widths"]
        for hand_t_grasp, gripper_width in zip(grasp_poses, gripper_widths):
            req = MoveitPlanningGripperPoseRequest()
            req.group_name = "left_arm"
            req.ntrial = 1
            req.gripper_pose = transform_matrix2pose(hand_t_grasp)
            req.gripper_pose.position.z -= 0.93
            req.gripper_link = "left_gripper"
            resp = _planning_with_gripper_pose_proxy(req)
            planning_result = resp.success
            planned_traj = resp.plan

            if planning_result:
                print("Planning to grasp succeeds.")
                planned_traj_list.append(planned_traj)
                pick_obj_idx = get_obj_idx_by_name(_next_object_list, _action['param'])

                support_obj_idx = get_obj_idx_by_name(_next_object_list, _next_object_list[pick_obj_idx].logical_state["on"][0])
                _next_object_list[support_obj_idx].logical_state["support"].remove(_next_object_list[pick_obj_idx].name)

                _next_object_list[pick_obj_idx].logical_state.clear()
                _next_object_list[pick_obj_idx].logical_state["held"] = []

                update_logical_state(_next_object_list)

                _after_grasp_left_joint_values = dict(zip(planned_traj.joint_names, planned_traj.points[-1].positions))
                synchronize_planning_scene(_after_grasp_left_joint_values, gripper_width, _right_joint_values,
                                           _right_gripper_width,
                                           _next_object_list, _meshes,
                                           _get_planning_scene_proxy=_get_planning_scene_proxy,
                                           _apply_planning_scene_proxy=_apply_planning_scene_proxy)

                req = MoveitPlanningJointValuesRequest()
                req.group_name = "left_arm"
                req.ntrial = 1
                req.joint_names = list(_init_left_joint_values.keys())
                req.joint_poses = list(_init_left_joint_values.values())

                resp = _planning_with_arm_joints_proxy(req)
                retreat_planning_result = resp.success
                retreat_planned_traj = resp.plan

                if retreat_planning_result:
                    print("Planning to retreat succeeds.")
                    planned_traj_list.append(retreat_planned_traj)
                    _after_retreat_left_joint_values = dict(zip(retreat_planned_traj.joint_names, retreat_planned_traj.points[-1].positions))
                    synchronize_planning_scene(_after_retreat_left_joint_values, gripper_width, _right_joint_values,
                                               _right_gripper_width,
                                               _next_object_list, _meshes,
                                               _get_planning_scene_proxy=_get_planning_scene_proxy,
                                               _apply_planning_scene_proxy=_apply_planning_scene_proxy)

                    resp = _get_planning_scene_proxy(GetPlanningSceneRequest())
                    current_scene = resp.scene
                    rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
                    rel_T = pose2transform_matrix(rel_obj_pose)

                    req = GetPositionFKRequest()
                    req.fk_link_names = ['left_gripper']
                    req.header.frame_id = 'world'
                    req.robot_state = current_scene.robot_state
                    resp = _compute_fk_proxy(req)

                    gripper_T = pose2transform_matrix(resp.pose_stamped[0].pose)
                    gripper_T[2, 3] += 0.93
                    _next_object_list[pick_obj_idx].pose = gripper_T.dot(rel_T)

                    return _next_object_list, _after_retreat_left_joint_values, gripper_width, _right_joint_values, _right_gripper_width, planned_traj_list

    if _action["type"] is "place":
        held_obj_idx = get_held_object(_next_object_list)

        resp = _get_planning_scene_proxy(GetPlanningSceneRequest())
        current_scene = resp.scene
        rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
        rel_T = pose2transform_matrix(rel_obj_pose)

        place_pose = _action["placing_pose"]
        gripper_pose = place_pose.dot(np.linalg.inv(rel_T))

        req = MoveitPlanningGripperPoseRequest()
        req.group_name = "left_arm"
        req.ntrial = 1
        req.gripper_pose = transform_matrix2pose(gripper_pose)
        req.gripper_pose.position.z -= 0.93
        req.gripper_link = "left_gripper"
        resp = _planning_with_gripper_pose_proxy(req)
        planning_result = resp.success
        planned_traj = resp.plan

        if planning_result:
            print("Planning to place succeeds.")
            planned_traj_list.append(planned_traj)
            _after_place_left_joint_values = dict(zip(planned_traj.joint_names, planned_traj.points[-1].positions))

            _next_object_list[held_obj_idx].pose = place_pose
            _next_object_list[held_obj_idx].logical_state.clear()
            _next_object_list[held_obj_idx].logical_state["on"] = [_next_object_list[held_obj_idx].name]
            _next_object_list[held_obj_idx].logical_state["done"] = []
            update_logical_state(_next_object_list)

            synchronize_planning_scene(_after_place_left_joint_values, _init_left_gripper_width,
                                       _right_joint_values,
                                       _right_gripper_width,
                                       _next_object_list, _meshes,
                                       _get_planning_scene_proxy=_get_planning_scene_proxy,
                                       _apply_planning_scene_proxy=_apply_planning_scene_proxy)

            req = MoveitPlanningJointValuesRequest()
            req.group_name = "left_arm"
            req.ntrial = 1
            req.joint_names = list(_init_left_joint_values.keys())
            req.joint_poses = list(_init_left_joint_values.values())

            resp = _planning_with_arm_joints_proxy(req)
            retreat_planning_result = resp.success
            retreat_planned_traj = resp.plan

            if retreat_planning_result:
                print("Planning to retreat succeeds.")
                planned_traj_list.append(retreat_planned_traj)
                _after_retreat_left_joint_values = dict(zip(retreat_planned_traj.joint_names, retreat_planned_traj.points[-1].positions))
                synchronize_planning_scene(_after_retreat_left_joint_values, _init_left_gripper_width, _right_joint_values,
                                           _right_gripper_width,
                                           _next_object_list, _meshes,
                                           _get_planning_scene_proxy=_get_planning_scene_proxy,
                                           _apply_planning_scene_proxy=_apply_planning_scene_proxy)
                return _next_object_list, _after_retreat_left_joint_values, _init_left_gripper_width, _right_joint_values, _right_gripper_width, planned_traj_list
    return None, None, None, None, None, None


def get_transition_with_robot(_object_list,
                              _left_joint_values, _left_gripper_width,
                              _right_joint_values, _right_gripper_width,
                              _action, _meshes,
                              _get_planning_scene_proxy=None,
                              _apply_planning_scene_proxy=None,
                              _planning_with_gripper_pose_proxy=None,
                              _planning_with_arm_joints_proxy=None,
                              _compute_fk_proxy=None,
                              _init_left_joint_values=None,
                              _init_left_gripper_width=0.03,
                              _init_right_joint_values=None,
                              _init_right_gripper_width=0.03):

    if _init_right_joint_values is None:
        _init_right_joint_values = {'right_w0': -0.6699952259595108,
                                    'right_w1': 1.030009435085784,
                                    'right_w2': 0.4999997247485215,
                                    'right_e0': 1.189968899785275,
                                    'right_e1': 1.9400238130755056,
                                    'right_s0': -0.08000397926829805,
                                    'right_s1': -0.9999781166910306}
    if _init_left_joint_values is None:
        _init_left_joint_values = {'left_w0': 0.6699952259595108,
                                   'left_w1': 1.030009435085784,
                                   'left_w2': -0.4999997247485215,
                                   'left_e0': -1.189968899785275,
                                   'left_e1': 1.9400238130755056,
                                   'left_s0': -0.08000397926829805,
                                   'left_s1': -0.9999781166910306}

    synchronize_planning_scene(_left_joint_values, _left_gripper_width, _right_joint_values, _right_gripper_width,
                               _object_list, _meshes,
                               _get_planning_scene_proxy=_get_planning_scene_proxy,
                               _apply_planning_scene_proxy=_apply_planning_scene_proxy)

    _next_object_list = deepcopy(_object_list)
    planned_traj_list = []
    if _action["type"] is "pick":

        hand_t_grasp = _action["grasp_pose"]
        # hand_t_retreat = _action["retreat_pose"]
        gripper_width = _action["gripper_width"]

        req = MoveitPlanningGripperPoseRequest()
        req.group_name = "left_arm"
        req.ntrial = 1
        req.gripper_pose = transform_matrix2pose(hand_t_grasp)
        req.gripper_pose.position.z -= 0.93
        req.gripper_link = "left_gripper"
        resp = _planning_with_gripper_pose_proxy(req)
        planning_result = resp.success
        planned_traj = resp.plan

        if planning_result:
            print("Planning to grasp succeeds.")
            planned_traj_list.append(planned_traj)
            pick_obj_idx = get_obj_idx_by_name(_next_object_list, _action['param'])

            support_obj_idx = get_obj_idx_by_name(_next_object_list, _next_object_list[pick_obj_idx].logical_state["on"][0])
            _next_object_list[support_obj_idx].logical_state["support"].remove(_next_object_list[pick_obj_idx].name)

            _next_object_list[pick_obj_idx].logical_state.clear()
            _next_object_list[pick_obj_idx].logical_state["held"] = []

            update_logical_state(_next_object_list)

            _after_grasp_left_joint_values = dict(zip(planned_traj.joint_names, planned_traj.points[-1].positions))
            synchronize_planning_scene(_after_grasp_left_joint_values, gripper_width, _right_joint_values,
                                       _right_gripper_width,
                                       _next_object_list, _meshes,
                                       _get_planning_scene_proxy=_get_planning_scene_proxy,
                                       _apply_planning_scene_proxy=_apply_planning_scene_proxy)

            req = MoveitPlanningJointValuesRequest()
            req.group_name = "left_arm"
            req.ntrial = 1
            req.joint_names = list(_init_left_joint_values.keys())
            req.joint_poses = list(_init_left_joint_values.values())

            resp = _planning_with_arm_joints_proxy(req)
            retreat_planning_result = resp.success
            retreat_planned_traj = resp.plan

            if retreat_planning_result:
                print("Planning to retreat succeeds.")
                planned_traj_list.append(retreat_planned_traj)
                _after_retreat_left_joint_values = dict(zip(retreat_planned_traj.joint_names, retreat_planned_traj.points[-1].positions))
                synchronize_planning_scene(_after_retreat_left_joint_values, gripper_width, _right_joint_values,
                                           _right_gripper_width,
                                           _next_object_list, _meshes,
                                           _get_planning_scene_proxy=_get_planning_scene_proxy,
                                           _apply_planning_scene_proxy=_apply_planning_scene_proxy)

                resp = _get_planning_scene_proxy(GetPlanningSceneRequest())
                current_scene = resp.scene
                rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
                rel_T = pose2transform_matrix(rel_obj_pose)

                req = GetPositionFKRequest()
                req.fk_link_names = ['left_gripper']
                req.header.frame_id = 'world'
                req.robot_state = current_scene.robot_state
                resp = _compute_fk_proxy(req)

                gripper_T = pose2transform_matrix(resp.pose_stamped[0].pose)
                gripper_T[2, 3] += 0.93
                _next_object_list[pick_obj_idx].pose = gripper_T.dot(rel_T)

                return _next_object_list, _after_retreat_left_joint_values, gripper_width, _right_joint_values, _right_gripper_width, planned_traj_list

    if _action["type"] is "place":
        held_obj_idx = get_held_object(_next_object_list)

        resp = _get_planning_scene_proxy(GetPlanningSceneRequest())
        current_scene = resp.scene
        rel_obj_pose = current_scene.robot_state.attached_collision_objects[0].object.mesh_poses[0]
        rel_T = pose2transform_matrix(rel_obj_pose)

        place_pose = _action["placing_pose"]
        gripper_pose = place_pose.dot(np.linalg.inv(rel_T))

        req = MoveitPlanningGripperPoseRequest()
        req.group_name = "left_arm"
        req.ntrial = 1
        req.gripper_pose = transform_matrix2pose(gripper_pose)
        req.gripper_pose.position.z -= 0.93
        req.gripper_link = "left_gripper"
        resp = _planning_with_gripper_pose_proxy(req)
        planning_result = resp.success
        planned_traj = resp.plan

        if planning_result:
            print("Planning to place succeeds.")
            planned_traj_list.append(planned_traj)
            _after_place_left_joint_values = dict(zip(planned_traj.joint_names, planned_traj.points[-1].positions))

            _next_object_list[held_obj_idx].pose = place_pose
            _next_object_list[held_obj_idx].logical_state.clear()
            _next_object_list[held_obj_idx].logical_state["on"] = [_next_object_list[held_obj_idx].name]
            _next_object_list[held_obj_idx].logical_state["done"] = []
            update_logical_state(_next_object_list)

            synchronize_planning_scene(_after_place_left_joint_values, _init_left_gripper_width,
                                       _right_joint_values,
                                       _right_gripper_width,
                                       _next_object_list, _meshes,
                                       _get_planning_scene_proxy=_get_planning_scene_proxy,
                                       _apply_planning_scene_proxy=_apply_planning_scene_proxy)

            req = MoveitPlanningJointValuesRequest()
            req.group_name = "left_arm"
            req.ntrial = 1
            req.joint_names = list(_init_left_joint_values.keys())
            req.joint_poses = list(_init_left_joint_values.values())

            resp = _planning_with_arm_joints_proxy(req)
            retreat_planning_result = resp.success
            retreat_planned_traj = resp.plan

            if retreat_planning_result:
                print("Planning to retreat succeeds.")
                planned_traj_list.append(retreat_planned_traj)
                _after_retreat_left_joint_values = dict(zip(retreat_planned_traj.joint_names, retreat_planned_traj.points[-1].positions))
                synchronize_planning_scene(_after_retreat_left_joint_values, _init_left_gripper_width, _right_joint_values,
                                           _right_gripper_width,
                                           _next_object_list, _meshes,
                                           _get_planning_scene_proxy=_get_planning_scene_proxy,
                                           _apply_planning_scene_proxy=_apply_planning_scene_proxy)
                return _next_object_list, _after_retreat_left_joint_values, _init_left_gripper_width, _right_joint_values, _right_gripper_width, planned_traj_list
    return None, None, None, None, None, None


if __name__ == '__main__':

    planning_without_robot = False
    check_meshes = True

    if planning_without_robot:
        mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
            configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points, goal_name='tower_goal')
        if check_meshes:
            visualize(initial_object_list, meshes, _goal_obj=goal_obj)

        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='twin_tower_goal')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)
        #
        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='box_goal')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)
        #
        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='stack_easy')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)
        #
        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='stack_difficult')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)

        action_list = get_possible_actions(initial_object_list, meshes, coll_mngr,
                                           contact_points, contact_faces, rotation_types)
        print(len(action_list))
        action = action_list[0]

        object_list = get_transition(initial_object_list, action)
        print(get_reward(initial_object_list, action, goal_obj, object_list, meshes))
        if check_meshes:
            visualize(object_list, meshes, _goal_obj=goal_obj)

        action_list = get_possible_actions(object_list, meshes, coll_mngr,
                                           contact_points, contact_faces, rotation_types)
        print(len(action_list))
        action = action_list[0]
        next_object_list = get_transition(object_list, action)
        print(get_reward(object_list, action, goal_obj, next_object_list, meshes))
        if check_meshes:
            visualize(next_object_list, meshes, _goal_obj=goal_obj)

        get_image(object_list, action, next_object_list, meshes, do_visualize=True)
    else:
        mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
            configuration_initializer(mesh_types, meshes, mesh_units, rotation_types, contact_faces, contact_points, goal_name='tower_goal')
        if check_meshes:
            visualize(initial_object_list, meshes, _goal_obj=goal_obj)

        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='twin_tower_goal')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)
        #
        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='box_goal')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)
        #
        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='stack_easy')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)
        #
        # mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
        # initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _, _ = \
        #     configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points, goal_name='stack_difficult')
        # visualize(initial_object_list, meshes, _goal_obj=goal_obj)

        initial_left_joint_values = {'left_w0': 0.6699952259595108,
                                     'left_w1': 1.030009435085784,
                                     'left_w2': -0.4999997247485215,
                                     'left_e0': -1.189968899785275,
                                     'left_e1': 1.9400238130755056,
                                     'left_s0': -0.08000397926829805,
                                     'left_s1': -0.9999781166910306}
        initial_left_gripper_width = 0.03
        initial_right_joint_values = {'right_w0': -0.6699952259595108,
                                      'right_w1': 1.030009435085784,
                                      'right_w2': 0.4999997247485215,
                                      'right_e0': 1.189968899785275,
                                      'right_e1': 1.9400238130755056,
                                      'right_s0': -0.08000397926829805,
                                      'right_s1': -0.9999781166910306}
        initial_right_gripper_width = 0.03

        rospy.init_node('mcts_moveit_planner_unit_test', anonymous=True)
        rospy.wait_for_service('/get_planning_scene')
        rospy.wait_for_service('/apply_planning_scene')
        rospy.wait_for_service('/compute_ik')
        rospy.wait_for_service('/compute_fk')

        get_planning_scene_proxy = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        apply_planning_scene_proxy = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
        compute_fk_proxy = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        planning_with_gripper_pose_proxy = rospy.ServiceProxy('/planning_with_gripper_pose', MoveitPlanningGripperPose)
        planning_with_arm_joints_proxy = rospy.ServiceProxy('/planning_with_arm_joints', MoveitPlanningJointValues)

        action_list = get_possible_actions_with_robot(initial_object_list, meshes, coll_mngr,
                                           contact_points, contact_faces, rotation_types)
        print(len(action_list))
        for action in action_list:
            object_list, left_joint_values, left_gripper_width, right_joint_values, right_gripper_width, planned_traj_list = \
                get_transition_with_robot(initial_object_list, initial_left_joint_values, initial_left_gripper_width,
                                          initial_right_joint_values, initial_right_gripper_width,
                                          action, meshes,
                                          _get_planning_scene_proxy=get_planning_scene_proxy,
                                          _apply_planning_scene_proxy=apply_planning_scene_proxy,
                                          _compute_fk_proxy=compute_fk_proxy,
                                          _planning_with_gripper_pose_proxy=planning_with_gripper_pose_proxy,
                                          _planning_with_arm_joints_proxy=planning_with_arm_joints_proxy)
            if object_list:
                break

        print(get_reward(initial_object_list, action, goal_obj, object_list, meshes))
        if check_meshes:
            visualize(object_list, meshes, _goal_obj=goal_obj)

        action_list = get_possible_actions_with_robot(object_list, meshes, coll_mngr,
                                           contact_points, contact_faces, rotation_types)
        print(len(action_list))
        for action in action_list:
            next_object_list, next_left_joint_values, next_left_gripper_width, next_right_joint_values, next_right_gripper_width, next_planned_traj_list = \
                get_transition_with_robot(object_list, left_joint_values, left_gripper_width, right_joint_values, right_gripper_width,
                                          action, meshes,
                                          _get_planning_scene_proxy=get_planning_scene_proxy,
                                          _apply_planning_scene_proxy=apply_planning_scene_proxy,
                                          _compute_fk_proxy=compute_fk_proxy,
                                          _planning_with_gripper_pose_proxy=planning_with_gripper_pose_proxy,
                                          _planning_with_arm_joints_proxy=planning_with_arm_joints_proxy)
            if next_object_list:
                break
        print(get_reward(object_list, action, goal_obj, next_object_list, meshes))
        if check_meshes:
            visualize(next_object_list, meshes, _goal_obj=goal_obj)