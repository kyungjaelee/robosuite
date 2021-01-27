import numpy as np
import trimesh

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class CustomTableArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_friction=(1, 0.005, 0.0001),
        xml="arenas/custom_table.xml",
        mesh_file="/home/kj/robosuite/robosuite/models/assets/arenas/meshes/custom_table.stl",
        mesh_scale=0.01,
    ):
        with open(mesh_file, 'rb') as f:
            self.mesh = trimesh.load(f, "stl")
        self.mesh.apply_scale(mesh_scale)
        # self.mesh.apply_translation(-self.mesh.center_mass)
        self.table_full_size = self.mesh.bounds[1] - self.mesh.bounds[0]
        # print(self.mesh.bounds)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = np.array([0., 0., 0.])

        super().__init__(xml_path_completion(xml))
        self.floor = self.worldbody.find("./geom[@name='floor']")

    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset