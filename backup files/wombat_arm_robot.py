import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Wombat_arm(ManipulatorModel):
    """
    Wombat_arm is a 6-DOF parallel manipulator to be used for pick and place operation

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/wombat_arm/wombat_arm.xml"), idn=idn)
    
    # def set_base_xpos(self):
	   #  return np.array([0.9, 0.06, 1.6])

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return None#"Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "default_wombat_arm"

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 2.0))

    @property
    def bottom_offset(self):
        return np.array([-0.6, -0.10, -1.6])

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
