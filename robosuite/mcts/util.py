from copy import deepcopy
import numpy as np

import trimesh
import pyrender

from robosuite.mcts.transforms import *
from robosuite.utils import transform_utils as tf

from matplotlib import pyplot as plt


def get_meshes(_mesh_types=['arch_box',
                            'rect_box',
                            'square_box',
                            'half_cylinder_box',
                            'triangle_box',
                            'twin_tower_goal',
                            'tower_goal',
                            'box_goal',
                            'custom_table'],
               _mesh_files=['./robosuite/models/assets/objects/meshes/arch_box.stl',
                            './robosuite/models/assets/objects/meshes/rect_box.stl',
                            './robosuite/models/assets/objects/meshes/square_box.stl',
                            './robosuite/models/assets/objects/meshes/half_cylinder_box.stl',
                            './robosuite/models/assets/objects/meshes/triangle_box.stl',
                            './robosuite/models/assets/objects/meshes/twin_tower_goal.stl',
                            './robosuite/models/assets/objects/meshes/tower_goal.stl',
                            './robosuite/models/assets/objects/meshes/box_goal.stl',
                            './robosuite/models/assets/objects/meshes/custom_table.stl'],
               _mesh_units=[0.001, 0.001, 0.001, 0.001, 0.001, 0.0011, 0.0011, 0.0011, 0.01],
               _area_ths=0.003,
               _rotation_types=4):

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
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
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


def get_grasp_pose():
    # TO DO
    return None


def get_on_pose(_name1, _pnt1, _normal1, _rotation1, _pnt2, _normal2, _pose2, _coll_mngr):
    target_pnt = _pnt1 + 1e-10 * _normal1
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


def configuration_initializer(_mesh_types, _meshes, _rotation_types, _contact_faces, _contact_points,
                              goal_name='tower_goal'):
    _object_list = []
    _coll_mngr = trimesh.collision.CollisionManager()

    if 'tower_goal' in goal_name:
        n_obj_per_mesh_types = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    elif 'twin_tower_goal' in goal_name:
        n_obj_per_mesh_types=[0, 2, 2, 0, 2, 0, 0, 0, 0, 0]
    elif 'stack_easy' in goal_name:
        n_obj_per_mesh_types=[0, 3, 3, 0, 0, 0, 0, 0, 0, 0]
    elif 'stack_difficult' in goal_name:
        n_obj_per_mesh_types=[0, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    else:
        assert Exception('goal name is wrong!!!')

    table_spawn_position = [0.3, 0.2]
    table_spawn_bnd_size = 0.075

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
        goal_pose = np.array([0.4, 0.0, _meshes[table_mesh_idx].bounds[1][2] - _meshes[table_mesh_idx].bounds[0][2]
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

            done = False
            for i in new_obj_contact_indices:
                for j in range(rotation_types):
                    for k in table_contact_indices:
                        pnt1 = new_obj_contact_points[i]
                        normal1 = new_obj_contact_normals[i]

                        pnt2 = table_contact_points[k]
                        normal2 = table_contact_normals[k]

                        pose1 = get_on_pose(new_obj.name, pnt1, normal1, 2.*np.pi*j/_rotation_types, pnt2, normal2,
                                            table_T, _coll_mngr)
                        if pose1 is not None:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if done:
                new_obj.pose = pose1
            _object_list.append(new_obj)
            update_logical_state(_object_list)

    if _goal_obj is not None:
        transform_g_t = np.linalg.inv(table_obj.pose).dot(_goal_obj.pose)
        _contact_points[table_mesh_idx] = transform_g_t[:3, :3].dot(_contact_points[goal_mesh_idx].T).T \
                                         + transform_g_t[:3, 3]
        _contact_faces[table_mesh_idx] = _contact_faces[table_mesh_idx][:len(_contact_faces[goal_mesh_idx])]
        # this is the most tricky part...
    return _object_list, _goal_obj, _contact_points, _contact_faces, _coll_mngr, table_spawn_position, table_spawn_bnd_size


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


def get_image(object_list, action, next_object_list, meshes=None, label=None, do_visualize=False):
    if meshes is None:
        meshes, mesh_types, mesh_files, mesh_units = get_meshes()

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


if __name__ == '__main__':
    mesh_types, mesh_files, mesh_units, meshes, rotation_types, contact_faces, contact_points = get_meshes()
    initial_object_list, goal_obj, contact_points, contact_faces, coll_mngr, _, _ = \
        configuration_initializer(mesh_types, meshes, rotation_types, contact_faces, contact_points)
    visualize(initial_object_list, meshes, _goal_obj=goal_obj)
