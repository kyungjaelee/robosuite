from copy import deepcopy
import numpy as np

import trimesh
import pyrender

from robosuite.mcts.transforms import *
from robosuite.utils import transform_utils as tf

from matplotlib import pyplot as plt


class Object(object):
    def __init__(self, name, mesh_idx, pose, logical_state):
        self.name = name
        self.mesh_idx = mesh_idx
        self.pose = pose
        self.logical_state = logical_state


def get_obj_idx_by_name(object_list, name):
    for obj_idx, obj in enumerate(object_list):
        if obj.name == name:
            return obj_idx
    return None


def get_held_object(object_list):
    for obj_idx, obj in enumerate(object_list):
        if "held" in obj.logical_state:
            return obj_idx
    return None


def update_logical_state(object_list):
    for obj in object_list:
        if "on" in obj.logical_state:
            for support_obj_name in obj.logical_state["on"]:
                support_obj_idx = get_obj_idx_by_name(object_list, support_obj_name)
                if "support" in object_list[support_obj_idx].logical_state:
                    if obj.name in object_list[support_obj_idx].logical_state["support"]:
                        continue
                    else:
                        object_list[support_obj_idx].logical_state["support"].append(obj.name)
                else:
                    object_list[support_obj_idx].logical_state["support"] = [obj.name]
        if "support" in obj.logical_state:
            for on_obj_name in obj.logical_state["support"]:
                on_obj_idx = get_obj_idx_by_name(object_list, on_obj_name)
                if "on" in object_list[on_obj_idx].logical_state:
                    if obj.name in object_list[on_obj_idx].logical_state["on"]:
                        continue
                    else:
                        object_list[on_obj_idx].logical_state["on"].append(obj.name)
                else:
                    object_list[on_obj_idx].logical_state["on"] = [obj.name]


def create_random_rotation_mtx_from_x(x_axis):
    rnd_axis = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    rnd_axis = rnd_axis / np.sqrt(np.sum(rnd_axis ** 2))

    y_axis = np.cross(rnd_axis, x_axis)
    y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.sqrt(np.sum(z_axis ** 2))

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis

    return T


def create_random_rotation_mtx_from_y(y_axis):
    rnd_axis = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    rnd_axis = rnd_axis / np.sqrt(np.sum(rnd_axis ** 2))

    z_axis = np.cross(rnd_axis, y_axis)
    z_axis = z_axis / np.sqrt(np.sum(z_axis ** 2))
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis

    return T


def create_random_rotation_mtx_from_z(z_axis):
    rnd_axis = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    rnd_axis = rnd_axis / np.sqrt(np.sum(rnd_axis ** 2))

    x_axis = np.cross(rnd_axis, z_axis)
    x_axis = x_axis / np.sqrt(np.sum(x_axis ** 2))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis

    return T


def sample_point_in_surface(mesh, face_idx):
    probs = np.random.uniform(size=(3, 1))
    probs = probs/np.sum(probs)
    return np.sum(probs*mesh.vertices[mesh.faces[face_idx]], axis=0)


def sample_initial_pose(obj1, obj2, _meshes, _coll_mngr, spawn_pose=[0.5, 0.], bnd_size=0.3):
    mesh1 = _meshes[obj1.mesh_idx]
    T1 = obj1.pose
    mesh2 = _meshes[obj2.mesh_idx]
    if 'arch_box' not in obj2.name:
        T2 = obj2.pose.dot(rotation_matrix(np.pi / 2., [0., 1., 0.], [0., 0., 0.]))
    else:
        T2 = obj2.pose
    while True:
        while True:
            normals1_world = T1[:3, :3].dot(mesh1.face_normals.T).T
            candidate_face_indices = np.where(normals1_world[:, 2] > np.cos(np.pi / 10.))[0]
            face_idx = np.random.choice(candidate_face_indices, 1)[0]

            pnt1 = sample_point_in_surface(mesh1, face_idx)
            normal1 = mesh1.face_normals[face_idx]
            normal1_world = T1[:3, :3].dot(normal1)
            pnt1_world = T1[:3, :3].dot(pnt1) + T1[:3, 3]
            if normal1_world[2] > np.cos(np.pi / 10.) and np.abs(spawn_pose[0] - pnt1_world[0]) < bnd_size and \
                    np.abs(spawn_pose[1] - pnt1_world[1]) < bnd_size:
                break

        while True:
            normals2_world = T2[:3, :3].dot(mesh2.face_normals.T).T
            candidate_face_indices = np.where(-normals2_world[:, 2] > np.cos(np.pi / 10.))[0]
            if len(candidate_face_indices) == 0:
                face_idx = np.random.choice(mesh2.face_normals.shape[0], 1)[0]
            else:
                face_idx = np.random.choice(candidate_face_indices, 1)[0]
            # print(normals2_world[face_idx])

            pnt2 = sample_point_in_surface(mesh2, face_idx)
            normal2 = mesh2.face_normals[face_idx]
            normal2_world = T2[:3, :3].dot(normal2)
            # print(normal2_world)
            if len(candidate_face_indices) == 0 or -normal2_world[2] > np.cos(np.pi / 10.):
                break

        target_pnts = pnt1 + 1e-10 * normal1
        target_normals = -normal1
        T_target = create_random_rotation_mtx_from_z(target_normals)
        T_target[:3, 3] = target_pnts

        T_source = create_random_rotation_mtx_from_z(normal2)
        T_source[:3, 3] = pnt2

        T2_new = T1.dot(T_target.dot(np.linalg.inv(T_source)))

        _coll_mngr.set_transform(obj2.name, T2_new)
        if not _coll_mngr.in_collision_internal():
            return T2_new
    return None


# def sample_on_pose_in_goal(goal_obj, obj1, meshes, coll_mngr, n_sampling_trial=100):
#     goal_mesh = meshes[goal_obj.mesh_idx]
#     T1 = goal_obj.pose
#     goal_normals_world = -T1[:3, :3].dot(goal_mesh.face_normals.T).T
#     candidate_face_indices = np.where(goal_normals_world[:, 2] > np.cos(np.pi / 10.))[0]
#     face_idx = np.random.choice(candidate_face_indices, 1)[0]
#
#     pnt1 = sample_point_in_surface(goal_mesh, face_idx)
#     normal1 = -goal_mesh.face_normals[face_idx]


# def sample_on_pose_in_goal(table_obj, obj2, goal_obj, meshes, coll_mngr, n_sampling_trial=100):
#     table_mesh = deepcopy(meshes[table_obj.mesh_idx])
#     table_mesh.apply_transform(table_obj.pose)
#
#     goal_mesh = deepcopy(meshes[goal_obj.mesh_idx])
#     goal_mesh.apply_transform(goal_obj.pose)
#     sampling_bounds = goal_mesh.bounds
#     mesh2 = meshes[obj2.mesh_idx]
#     for _ in range(n_sampling_trial):
#         sampled_point = (sampling_bounds[1] - sampling_bounds[0])*np.random.uniform(size=3) + sampling_bounds[0]
#
#
#         pnt1 = sample_point_in_surface(mesh1, face_idx)
#         normal1 = mesh1.face_normals[face_idx]
#
#         pnts2, tri2_idices = mesh2.sample(1, return_index=True)
#         normals2 = mesh2.face_normals[tri2_idices]
#
#         target_pnts = pnt1 + 1e-10 * normal1
#         target_normals = -normal1
#         T_target = create_random_rotation_mtx_from_z(target_normals)
#         T_target[:3, 3] = target_pnts
#
#         T_source = create_random_rotation_mtx_from_z(normals2[0])
#         T_source[:3, 3] = pnts2
#
#         T2 = T1.dot(T_target.dot(np.linalg.inv(T_source)))
#
#         coll_mngr.set_transform(obj2.name, T2)
#         if not coll_mngr.in_collision_internal():
#             return T2
#     return None

def critical_vertices_detector(_mesh):
    critical_edges = _mesh.face_adjacency_edges[(_mesh.face_adjacency_angles > np.pi/2.5) * _mesh.face_adjacency_convex]
    critical_vertices = np.unique(critical_edges.reshape((-1,)))
    return critical_edges, critical_vertices


def sample_critical_contact_frame(_mesh):
    critical_edges, critical_vertices = critical_vertices_detector(_mesh)
    critical_vertex = np.random.choice(critical_vertices)
    T = np.eye(4)
    while True:
        face_indices = _mesh.vertex_faces[critical_vertex]
        f1, f2 = np.random.choice(face_indices[face_indices > 0], size=2)
        x_axis = _mesh.face_normals[f1]
        z_axis = _mesh.face_normals[f2]
        y_axis = np.cross(z_axis, x_axis)
        if np.sum(y_axis ** 2) > 0.:
            y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
            T[:3, 0] = x_axis
            T[:3, 1] = y_axis
            T[:3, 2] = z_axis
            break
        else:
            continue

    T[:3, 3] = _mesh.vertices[critical_vertex]
    return T


def sample_pose_in_goal_mesh(table_obj, obj2, goal_obj, meshes, coll_mngr, n_sampling_trial=10):
    goal_mesh = deepcopy(meshes[goal_obj.mesh_idx])
    goal_mesh = goal_mesh.apply_transform(goal_obj.pose)
    table_mesh = deepcopy(meshes[table_obj.mesh_idx])
    table_mesh = table_mesh.apply_transform(table_obj.pose)
    mesh2 = meshes[obj2.mesh_idx]

    for _ in range(n_sampling_trial):
        T_target = sample_critical_contact_frame(goal_mesh)
        T_source = sample_critical_contact_frame(mesh2)
        T2 = T_target.dot(np.linalg.inv(T_source))
        coll_mngr.set_transform(obj2.name, T2)
        if not coll_mngr.in_collision_internal():
            return T2
    return None


def sample_on_pose_with_bias(obj1, obj2, goal_obj, meshes, coll_mngr, n_sampling_trial=10, goal_unbias=0.3):
    mesh1 = deepcopy(meshes[obj1.mesh_idx])
    T1 = obj1.pose
    normals1_world = T1[:3, :3].dot(mesh1.face_normals.T).T
    candidate_face_indices = np.where(normals1_world[:, 2] > np.cos(np.pi / 10.))[0]

    goal_mesh = deepcopy(meshes[goal_obj.mesh_idx])
    goal_mesh.apply_transform(goal_obj.pose)
    mesh2 = meshes[obj2.mesh_idx]
    if len(candidate_face_indices) == 0:
        return None
    else:
        for _ in range(n_sampling_trial):
            if np.random.uniform() > goal_unbias:
                sampled_point = []
                face_indices = []
                for face_idx in candidate_face_indices:
                    for _ in range(10000):
                        sampled_point.append(sample_point_in_surface(mesh1, face_idx))
                        face_indices.append(face_idx)
                sampled_point = np.asarray(sampled_point)
                sampled_point_world = (T1[:3, :3].dot(sampled_point.T) + T1[:3, [3]]).T
                distance_to_goal_mesh = goal_mesh.nearest.signed_distance(sampled_point_world)
                face_idx = face_indices[np.argmax(distance_to_goal_mesh)]
            else:
                face_idx = np.random.choice(candidate_face_indices, 1)[0]

                pnt1 = sample_point_in_surface(mesh1, face_idx)
                normal1 = mesh1.face_normals[face_idx]

                pnts2, tri2_idices = mesh2.sample(1, return_index=True)
                normals2 = mesh2.face_normals[tri2_idices]

                target_pnts = pnt1 + 1e-10 * normal1
                target_normals = -normal1
                T_target = create_random_rotation_mtx_from_z(target_normals)
                T_target[:3, 3] = target_pnts

                T_source = create_random_rotation_mtx_from_z(normals2[0])
                T_source[:3, 3] = pnts2

                T2 = T1.dot(T_target.dot(np.linalg.inv(T_source)))

                coll_mngr.set_transform(obj2.name, T2)
                if not coll_mngr.in_collision_internal():
                    return T2
        return None


def sample_on_pose(obj1, obj2, meshes, coll_mngr, n_sampling_trial=10):
    mesh1 = meshes[obj1.mesh_idx]
    T1 = obj1.pose
    mesh2 = meshes[obj2.mesh_idx]
    normals1_world = T1[:3, :3].dot(mesh1.face_normals.T).T
    candidate_face_indices = np.where(normals1_world[:, 2] > np.cos(np.pi / 10.))[0]
    if len(candidate_face_indices) == 0:
        return None
    else:
        for _ in range(n_sampling_trial):
            face_idx = np.random.choice(candidate_face_indices, 1)[0]

            pnt1 = sample_point_in_surface(mesh1, face_idx)
            normal1 = mesh1.face_normals[face_idx]

            pnts2, tri2_idices = mesh2.sample(1, return_index=True)
            normals2 = mesh2.face_normals[tri2_idices]

            target_pnts = pnt1 + 1e-10 * normal1
            target_normals = -normal1
            T_target = create_random_rotation_mtx_from_z(target_normals)
            T_target[:3, 3] = target_pnts

            T_source = create_random_rotation_mtx_from_z(normals2[0])
            T_source[:3, 3] = pnts2

            T2 = T1.dot(T_target.dot(np.linalg.inv(T_source)))

            coll_mngr.set_transform(obj2.name, T2)
            if not coll_mngr.in_collision_internal():
                return T2
        return None


def sample_grasp_pose(obj1, meshes, n_sampling_trial=100):
    mesh1 = meshes[obj1.mesh_idx]
    T1 = obj1.pose
    for _ in range(n_sampling_trial):
        pnts1, tri1_idices = mesh1.sample(1, return_index=True)
        normals1 = mesh1.face_normals[tri1_idices]

        locations, index_ray, index_tri = mesh1.ray.intersects_location(
            ray_origins=pnts1 - 1e-5 * normals1,
            ray_directions=-normals1)

        pnts2 = locations[0]
        c_pnt = (pnts1 + pnts2) / 2.

        y_axis = pnts2 - pnts1
        dist = np.sqrt(np.sum(y_axis ** 2))
        y_axis = y_axis / np.sqrt(np.sum(y_axis ** 2))
        T_grasp = create_random_rotation_mtx_from_y(y_axis)
        T_grasp[:3, 3] = c_pnt

        T_retreat = deepcopy(T_grasp)
        approaching_dir = T_retreat[:3, 2]
        T_retreat[:3, 3] = c_pnt - 8e-2 * approaching_dir

        hand_T_grasp = T1.dot(T_grasp)
        hand_T_retreat = T1.dot(T_retreat)
        if hand_T_grasp[2, 2] < -0.7:
            return hand_T_grasp, hand_T_retreat, T_grasp, dist / 2.
    return None


def get_meshes():
    mesh_types = ['arch_box',
                  'rect_box',
                  'square_box',
                  'half_cylinder_box',
                  'triangle_box',
                  'twin_tower_goal',
                  'box_goal',
                  'custom_table']
    mesh_files = ['/home/kj/robosuite/robosuite/models/assets/objects/meshes/arch_box.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/rect_box.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/square_box.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/half_cylinder_box.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/triangle_box.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/twin_tower_goal.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/box_goal.stl',
                  '/home/kj/robosuite/robosuite/models/assets/objects/meshes/custom_table.stl']
    mesh_units = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
    n_mesh_types = len(mesh_types)

    _meshes = []
    for mesh_type, mesh_file, unit in zip(mesh_types, mesh_files, mesh_units):
        mesh = trimesh.load(mesh_file)

        # clean mesh vertices and surfaces
        # mesh.remove_degenerate_faces()
        # mesh.remove_unreferenced_vertices()
        # mesh.remove_infinite_values()
        # mesh.remove_duplicate_faces()

        # centering a mesh
        mesh.apply_scale(unit)
        mesh.apply_translation(-mesh.center_mass)

        # set random stable pose
        if mesh_type is not "custom_table":
            stable_poses, probs = mesh.compute_stable_poses()
            stable_pose_idx = np.argmax(probs)
            # stable_pose = stable_poses[stable_pose_idx]
            # mesh.apply_transform(stable_pose)

        _meshes.append(mesh)
    return _meshes, mesh_types, mesh_files, mesh_units


def table_objects_initializer(n_obj_per_mesh_types=[0, 2, 2, 2, 2, 0, 0, 0], spawn_pose=[0.5, 0.], bnd_size=0.2, random_initial=True):
    _meshes, mesh_types, mesh_files, mesh_units = get_meshes()
    _object_list = []
    _coll_mngr = trimesh.collision.CollisionManager()

    goal_name = 'box_goal'
    goal_pose = np.array([0.4, 0.0, 0.573077+0.211923])
    goal_mesh_idx = 6
    goal_T = np.eye(4)
    goal_T[:3, 3] = goal_pose
    _goal_obj = Object(goal_name, goal_mesh_idx, goal_T, {"goal": []})

    table_name = 'custom_table'
    table_pose = np.array([0.6, 0.0, 0.573077])
    table_mesh_idx = 7
    table_T = np.eye(4)
    table_T[:3, 3] = table_pose
    table_top_z_offset_from_com = np.max(_meshes[table_mesh_idx].bounds, axis=0)[2]
    table_obj = Object(table_name, table_mesh_idx, table_T, {"static": []})

    _object_list.append(table_obj)
    _coll_mngr.add_object(table_name, _meshes[table_mesh_idx])
    _coll_mngr.set_transform(table_name, table_T)

    for mesh_idx, n_obj in enumerate(n_obj_per_mesh_types):
        obj_mesh_idx = mesh_idx
        for obj_idx in range(n_obj):
            obj_name = mesh_types[mesh_idx] + str(obj_idx)

            if "custom_table" not in mesh_types[mesh_idx]:
                stable_poses, probs = _meshes[mesh_idx].compute_stable_poses(n_samples=100)
                stable_pose_idx = np.argmax(probs)
                stable_pose = stable_poses[stable_pose_idx]
                if "half_cylinder_box" in mesh_types[mesh_idx] or "triangle_box" in mesh_types[mesh_idx]:
                    stable_pose = tf.rotation_matrix(-np.pi/2., [1., 0., 0.]).dot(stable_pose)
            else:
                stable_pose = np.eye(4)

            new_obj = Object(obj_name, obj_mesh_idx, stable_pose, {"on": [table_name]})

            _coll_mngr.add_object(obj_name, _meshes[obj_mesh_idx])
            if random_initial:
                new_obj.pose = sample_initial_pose(table_obj, new_obj, _meshes, _coll_mngr, spawn_pose, bnd_size)
            else:
                new_obj.pose[:3, 3] = table_pose
                new_obj.pose[2, 3] += 0.17 + 0.04*obj_idx + 0.04*np.sum(n_obj_per_mesh_types[:mesh_idx])
            _object_list.append(new_obj)
            update_logical_state(_object_list)
    return _object_list, _meshes, _coll_mngr, _goal_obj


def visualize(_object_list, _meshes):
    trimesh_scene = trimesh.Scene()
    for obj in _object_list:
        trimesh_scene.add_geometry(_meshes[obj.mesh_idx], node_name=obj.name, transform=obj.pose)
    trimesh_scene.show()


def get_possible_actions(object_list):
    action_list = []

    # Check pick
    if all(["held" not in obj.logical_state and "contact" not in obj.logical_state for obj in object_list]):
        for obj in object_list:
            if "support" not in obj.logical_state and "static" not in obj.logical_state and "prev_pick" not in obj.logical_state:
                action_list.append({"type": "pick", "param": obj.name})

    # Check place
    if any(["held" in obj.logical_state for obj in object_list]):
        for obj in object_list:
            if "held" not in obj.logical_state:
                action_list.append({"type": "place", "param": obj.name})

    return action_list


def get_possible_actions_with_bias(object_list, goal_obj, meshes):
    action_list = []

    # Check pick
    if all(["held" not in obj.logical_state and "contact" not in obj.logical_state for obj in object_list]):
        for obj in object_list:
            if "support" not in obj.logical_state and "static" not in obj.logical_state and "prev_pick" not in obj.logical_state:
                action_list.append({"type": "pick", "param": obj.name})

    # Check place
    if any(["held" in obj.logical_state for obj in object_list]):
        goal_mesh = deepcopy(meshes[goal_obj.mesh_idx])
        goal_mesh = goal_mesh.apply_transform(goal_obj.pose)
        for obj in object_list:
            obj_mesh = deepcopy(meshes[obj.mesh_idx])
            obj_mesh = obj_mesh.apply_transform(obj.pose)
            iou = goal_mesh.intersection(obj_mesh).volume/obj_mesh.volume
            if "held" not in obj.logical_state and (iou > 0. or "static" in obj.logical_state):
                action_list.append({"type": "place", "param": obj.name})

    return action_list


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
    object_list, meshes, coll_mngr, goal_obj = table_objects_initializer()
    visualize(object_list, meshes)

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

    if action["type"] is "place":
        place_obj_idx = get_obj_idx_by_name(object_list, action['param'])
        held_obj_idx = get_held_object(object_list)
        object_list[held_obj_idx].pose = sample_on_pose(object_list[place_obj_idx], object_list[held_obj_idx], meshes,
                                                        coll_mngr)

        object_list[held_obj_idx].logical_state.clear()
        object_list[held_obj_idx].logical_state["on"] = [object_list[place_obj_idx].name]
        object_list[held_obj_idx].logical_state["prev_pick"] = []
        update_logical_state(object_list)

        for obj in object_list:
            print(obj.name, obj.logical_state)

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

    if action["type"] is "place":
        place_obj_idx = get_obj_idx_by_name(object_list, action['param'])
        held_obj_idx = get_held_object(object_list)
        object_list[held_obj_idx].pose = sample_on_pose(object_list[place_obj_idx], object_list[held_obj_idx], meshes,
                                                        coll_mngr)

        object_list[held_obj_idx].logical_state.clear()
        object_list[held_obj_idx].logical_state["on"] = [object_list[place_obj_idx].name]
        object_list[held_obj_idx].logical_state["prev_pick"] = []
        update_logical_state(object_list)

        for obj in object_list:
            print(obj.name, obj.logical_state)

