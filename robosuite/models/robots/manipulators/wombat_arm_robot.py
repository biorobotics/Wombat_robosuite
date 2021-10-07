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
		# print("init() wombat_arm_robot.py file")
	
	# def set_base_xpos(self):
	   #  return np.array([0.9, 0.06, 1.6])

	@property
	def default_mount(self):
		# print("default_mount() wombat_arm_robot.py file")
		return None

	@property
	def default_gripper(self):
		# print("default_gripper() wombat_arm_robot.py file")
		return None#"D3_gripper"

	@property
	def default_controller_config(self):
		# print("default_controller_config() wombat_arm_robot.py file")
		return "default_wombat_arm"

	@property
	def init_qpos(self):
		#return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
		# print("init_qpos() wombat_arm_robot.py file")
		return np.zeros(19)

	@property
	def base_xpos_offset(self):
		# print("base_xpos_offset() wombat_arm_robot.py file")
		return {
			"bins": (-0.5, -0.1, 0),
			"belts": (-0.5, -0.1, 0),
			"empty": (-0.6, 0, 0),
			"table": lambda table_length: (-0.16 - table_length/2, 0, 0)
		}

	@property
	def top_offset(self):
		# print("top_offset() wombat_arm_robot.py file")
		return np.array((0, 0, 2.0))

	@property
	def bottom_offset(self):
		# print("bottom_offset() wombat_arm_robot.py file")
		return np.array([-0.6, -0.10, -1.6])

	@property
	def _horizontal_radius(self):
	    return 0

	@property
	def arm_type(self):
		# print("arm_type() wombat_arm_robot.py file")
		return "single"
