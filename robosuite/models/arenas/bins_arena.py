import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string


class BinsArena(Arena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        bin1_pos (3-tuple): (x,y,z) position to place bin1
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self, bin1_pos=(0.1, -0.5, 0.8), Conveyor_belt1_pos=(0, 0, 0.4), table_full_size=(0.39, 0.49, 0.82), table_friction=(1, 0.005, 0.0001)
    ):
        super().__init__(xml_path_completion("arenas/bins_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.bin1_body = self.worldbody.find("./body[@name='bin1']")
        self.bin2_body = self.worldbody.find("./body[@name='bin2']")
        self.table_top_abs = np.array(bin1_pos)

        self.configure_location()
        # self.change_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

    # def change_location(self):
    #     """change location at every time step"""
    #     bin1_pos=(0.1 + np.random.uniform(1,2), -0.5 + np.random.uniform(1,2), 0.8 + np.random.uniform(1,2)) 
    #     super().__init__(xml_path_completion("arenas/bins_arena.xml"))
