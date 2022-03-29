import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class UR3e(ManipulatorModel):
    """
    UR3e is a sleek and elegant new robot created by Universal Robots

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ur3e/ur3e.xml"), idn=idn)

    @property
    def default_mount(self):
        return None#"RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "default_ur3e"

    @property
    def init_qpos(self):
        return np.array([-np.pi/2, -2.0, -np.pi/2, -1.01,  1.57, np.pi *0/180.0])
        # return np.zeros(6)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def bottom_offset(self):
        # print("bottom_offset() wombat_arm_robot.py file")
        return np.array([0, 0, -0.75])

    @property
    def _horizontal_radius(self):
        return 0

    @property
    def arm_type(self):
        return "single"
