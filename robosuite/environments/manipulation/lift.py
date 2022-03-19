from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class Lift(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" paramhone_quat', 'gripper_to_phone_pos', 'robot0_proprio-state', 'object-state'])
odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'phone_pos', 'phone_quat', 'gripper_to_phone_pos', 'robot0_proprio-state', 'object-state'])
Pressed ESC
 factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
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

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

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
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.5, 5, 0.82),   #0.39, 0.49, 0.82 make this as same size as the whole conveyor 
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None, #either give a function that does the random sampling, randomized dynamics function part might go in here 
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera='None',       #setting this to 'None' lets you drag around the environment and pan around using the mouse
        render_collision_mesh=True, #Can set to false #TODO 
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=11000,               #set this param as same as the training script
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # self.table_offset = np.array((0.6, -2, 0))
        self.table_offset = np.array((0,0,0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
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
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted
            - #TODO add a partial reward if needed

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            phone_pos = self.sim.data.body_xpos[self.phone_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - phone_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.phone):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "iphone",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.phone = BoxObject(
            name = "iphone", 
            size = [0.039,0.08,0.0037],import collections
import numpy as np

from copy import copy

from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_multiply
from robosuite.models.objects import MujocoObject


class ObjectPositionSampler:
	"""
	Base class of object placement sampler.

	Args:
		name (str): Name of this sampler.

		mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

		ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
			(should be implemented by subclass)

		ensure_valid_placement (bool): If True, will check for correct (valid) object placements

		reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

		z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
			that do not move (i.e. no free joint) to place them above the table.
	"""

	def __init__(
		self,
		name,
		mujoco_objects=None,
		ensure_object_boundary_in_range=True,
		ensure_valid_placement=True,
		reference_pos=(0, 0, 0),
		z_offset=0.,
	):
		# Setup attributes
		self.name = name
		if mujoco_objects is None:
			self.mujoco_objects = []
		else:
			# Shallow copy the list so we don't modify the inputted list but still keep the object references
			self.mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else copy(mujoco_objects)
		self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
		self.ensure_valid_placement = ensure_valid_placement
		self.reference_pos = reference_pos
		self.z_offset = z_offset

	def add_objects(self, mujoco_objects):
		"""
		Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.

		Args:
			mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
		"""
		mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
		for obj in mujoco_objects:
			# assert obj not in self.mujoco_objects, "Object '{}' already in sampler!".format(obj.name)
			self.mujoco_objects.append(obj)

	def reset(self):
		"""
		Resets this sampler. Removes all mujoco objects from this sampler.
		"""
		self.mujoco_objects = []

	def sample(self, fixtures=None, reference=None, on_top=True):
		"""
		Uniformly sample on a surface (not necessarily table surface).

		Args:
			fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
				obstacles that should not be in contact with newly sampled objects. Used to make sure newly
				generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

			reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
				corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
				relative to this sampler's `'reference_pos'` value.

			on_top (bool): if True, sample placement on top of the reference object.

		Return:
			dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
				placements specified in @fixtures. Note quat is in (w,x,y,z) form
		"""
		raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
	"""
	Places all objects within the table uniformly random.

	Args:            object_placements['iphone'][0][0] = 0.2
            object_placements['iphone'][0][1] = -0.2
		mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

		x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

		y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

		rotation (None or float or Iterable):
			:`None`: Add uniform random random rotation
			:`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
			:`value`: Add fixed angle rotation

		rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

		ensure_object_boundary_in_range (bool):
			:`True`: The center of object is at position:
				 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
			:`False`:
				[uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

		ensure_valid_placement (bool): If True, will check for correct (valid) object placements

		reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

		z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
			that do not move (i.e. no free joint) to place them above the table.
	"""

	def __init__(
		self,
		name,
		mujoco_objects=None,
		x_range=(0, 0),
		y_range=(0, 0),
		rotation=None,
		rotation_axis='z',
		ensure_object_boundary_in_range=True,
		ensure_valid_placement=True,
		reference_pos=(0, 0, 0),
		z_offset=0.,
	):
		self.x_range = x_range
		self.y_range = y_range
		self.rotation = rotation
		self.rotation_axis = rotation_axis
		# print("x_range", x_range)
		# print("y_range", y_range)
		# print("_init_ in UniformRandomSampler")

		super().__init__(
			name=name,
			mujoco_objects=mujoco_objects,
			ensure_object_boundary_in_range=ensure_object_boundary_in_range,
			ensure_valid_placement=ensure_valid_placement,
			reference_pos=reference_pos,
			z_offset=z_offset,
		)

	def _sample_x(self, object_horizontal_radius):
		"""
		Samples the x location for a given object

		Args:
			object_horizontal_radius (float): Radius of the object currently being sampled for

		Returns:
			float: sampled x position
		"""
		# print("self.x_range", self.x_range)
		# print("_sample_x in UniformRandomSampler")
		minimum, maximum = self.x_range
		if self.ensure_object_boundary_in_range:
			minimum += object_horizontal_radius
			maximum -= object_horizontal_radius
		print(f"placement_samplers.py-> object_hori radius {object_horizontal_radius}, max: {maximum}, min {minimum}, value: {np.random.uniform(high=maximum, low=minimum)}")
		return np.random.uniform(high=maximum, low=minimum)

	def _sample_y(self, object_horizontal_radius):
		"""
		Samples the y location for a given object

		Args:
			object_horizontal_radius (float): Radius of the object currently being sampled for

		Returns:
			float: sampled y position
		"""
		# print("_sample_y in UniformRandomSampler")
		minimum, maximum = self.y_range
		if self.ensure_object_boundary_in_range:
			minimum += object_horizontal_radius
			maximum -= object_horizontal_radius
		return np.random.uniform(high=maximum, low=minimum)

	def _sample_quat(self):
		"""
		Samples the orientation for a given object

		Returns:
			np.array: sampled (r,p,y) euler angle orientation

		Raises:
			ValueError: [Invalid rotation axis]
		"""
		# print("_sample_quat in UniformRandomSampler")
		if self.rotation is None:
			rot_angle = np.random.uniform(high=2 * np.pi, low=0)
		elif isinstance(self.rotation, collections.Iterable):
			rot_angle = np.random.uniform(
				high=max(self.rotation), low=min(self.rotation)
			)
		else:
			rot_angle = self.rotation

		# Return angle based on axis requested
		if self.rotation_axis == 'x':
			return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
		elif self.rotation_axis == 'y':
			return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
		elif self.rotation_axis == 'z':
			return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
		else:
			# Invalid axis specified, raise error
			raise ValueError("Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis))

	def sample(self, fixtures=None, reference=None, on_top=True):
		"""
		Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

		Args:
			fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
				obstacles that should not be in contact with newly sampled objects. Used to make sure newly
				generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

			reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
				corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
				relative to this sampler's `'reference_pos'` value.

			on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
				z-offset of the current sampled object's bottom_offset + the reference object's top_offset
				(if specified)

		Return:
			dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
				placements specified in @fixtures. Note quat is in (w,x,y,z) form

		Raises:
			RandomizationError: [Cannot place all objects]
			AssertionError: [Reference object name does not exist, invalid inputs]
		"""
		# Standardize inputs
		# print("fixtures", fixtures)
		# a_file = open("test.txt", "w")
		# a_file.write(str(fixtures))
		# a_file.close()
		# fixtures = {'Milk': ((0.1, -1.75, 0.885), array([0.87616461, 0.        , 0.        , 0.48201201]), <robosuite.models.objects.xml_objects.MilkObject object at 0x7f6a00c12b38>), 'Bread': ((0.1, -1.75, 0.8450000000000001), array([0.30813623, 0.        , 0.        , 0.95134224]), <robosuite.models.objects.xml_objects.BreadObject object at 0x7f6a00b98c50>), 'Cereal': ((0.1, -1.75, 0.9), array([-0.96967114,  0.        ,  0.        ,  0.24441334]), <robosuite.models.objects.xml_objects.CerealObject object at 0x7f6a00b9c860>), 'Can': ((0.1, -1.75, 0.8600000000000001), array([0.51344005, 0.        , 0.        , 0.85812546]), <robosuite.models.objects.xml_objects.CanObject object at 0x7f6a00b9cc18>), 'iPhone12ProMax': ((0.1, -1.75, 0.8600000000000001), array([-0.93044896,  0.        ,  0.        ,  0.36642151]), <robosuite.models.objects.xml_objects.iPhone12ProMaxObject object at 0x7f6a00ba3828>)}
		# print("sample in UniformRandomSampler")
		placed_objects = {} if fixtures is None else copy(fixtures)
		if reference is None:
			base_offset = self.reference_pos
		elif type(reference) is str:
			assert reference in placed_objects, "Invalid reference received. Current options are: {}, requested: {}"\
				.format(placed_objects.keys(), reference)
			ref_pos, _, ref_obj = placed_objects[reference]
			base_offset = np.array(ref_pos)
			if on_top:
				base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
		else:
			base_offset = np.array(reference)
			assert base_offset.shape[0] == 3, "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}"\
				.format(base_offset)
		# print("sample")
		# Sample pos and quat for all objects assigned to this sampler
		# print("hello outside")
		# print("self.mujoco_objects", self.mujoco_objects)
		for obj in self.mujoco_objects:
			# First make sure the currently sampled object hasn't already been sampled
			assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)
			# print("hello inside")
			horizontal_radius = obj.horizontal_radius
			bottom_offset = obj.bottom_offset
			success = False

			for i in range(5000):  # 5000 retries
				object_x = self._sample_x(horizontal_radius) + base_offset[0] + np.random.randint(2)
				object_y = self._sample_y(horizontal_radius) + base_offset[1] + np.random.randint(2)
				object_z = self.z_offset + base_offset[2]
				print(f"placement_sampler.py ->  object_x {object_x}, object_y {object_y}")
				if on_top:
					object_z -= bottom_offset[-1]

				# objects cannot overlap
				location_valid = True
				if self.ensure_valid_placement:
					for (x, y, z), _, other_obj in placed_objects.values():
						if (
							np.linalg.norm((object_x - x, object_y - y))
							<= other_obj.horizontal_radius + horizontal_radius
						) and (
							object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
						):
							location_valid = False
							break

				if location_valid:
					# random rotation
					quat = self._sample_quat()

					# multiply this quat by the object's initial rotation if it has the attribute specified
					if hasattr(obj, "init_quat"):
						quat = quat_multiply(quat, obj.init_quat)

					# location is valid, put the object down
					# pos = (object_x, object_y, object_z)
					# pos = (0.145, 0.195, object_z)
					##custom code
					object_x = np.random.uniform(-0.125, 0.145)
					object_y = np.random.uniform(-0.195, 0.195)
					pos = (object_x, object_y, object_z)
					placed_objects[obj.name] = (pos, quat, obj)
					# print("pose in placement_samplers", pos)
					success = True
					break

			if not success:
			    raise RandomizationError("Cannot place all objects ):")
		# print("placed_objects in urs", placed_objects)
		return placed_objects


class SequentialCompositeSampler(ObjectPositionSampler):
	"""
	Samples position for each object sequentially. Allows chaining
	multiple placement initializers together - so that object locations can
	be sampled on top of other objects or relative to other object placements.

	Args:
		name (str): Name of this sampler.
	"""
	def __init__(self, name):
		# Samplers / args will be filled in later
		self.samplers = collections.OrderedDict()
		self.sample_args = collections.OrderedDict()
		# print("_init_ in SequentialCompositeSampler")
		super().__init__(name=name)

	def append_sampler(self, sampler, sample_args=None):
		"""
		Adds a new placement initializer with corresponding @sampler and arguments

		Args:
			sampler (ObjectPositionSampler): sampler to add
			sample_args (None or dict): If specified, should be additional arguments to pass to @sampler's sample()
				call. Should map corresponding sampler's arguments to values (excluding @fixtures argument)

		Raises:
			AssertionError: [Object name in samplers]
		"""
		# Verify that all added mujoco objects haven't already been added, and add to this sampler's objects dict
		# print("sampler_args", sample_args)
		# print("append_sampler in SequentialCompositeSampler")
		for obj in sampler.mujoco_objects:
			assert obj not in self.mujoco_objects, f"Object '{obj.name}' already has sampler associated with it!"
			self.mujoco_objects.append(obj)
		self.samplers[sampler.name] = sampler
		self.sample_args[sampler.name] = sample_args

	def hide(self, mujoco_objects):
		"""
		Helper method to remove an object from the workspace.

		Args:
			mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to hide
		"""
		# print("hide in SequentialCompositeSampler")
		sampler = UniformRandomSampler(
			name="HideSampler",
			mujoco_objects=mujoco_objects,
			x_range=[-10, -20],
			y_range=[-10, -20],
			rotation=[0, 0],
			rotation_axis='z',
			z_offset=10,
			ensure_object_boundary_in_range=False,
			ensure_valid_placement=False,
		)
		self.append_sampler(sampler=sampler)

	def add_objects(self, mujoco_objects):
		"""
		Override super method to make sure user doesn't call this (all objects should implicitly belong to sub-samplers)
		"""
		# print("add_objects in SequentialCompositeSampler")
		raise AttributeError("add_objects() should not be called for SequentialCompsiteSamplers!")

	def add_objects_to_sampler(self, sampler_name, mujoco_objects):
		"""
		Adds specified @mujoco_objects to sub-sampler with specified @sampler_name.

		Args:
			sampler_name (str): Existing sub-sampler name
			mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to add
		"""
		# First verify that all mujoco objects haven't already been added, and add to this sampler's objects dict
		# print("add_objects_to_sampler in SequentialCompositeSampler")
		mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
		for obj in mujoco_objects:
			assert obj not in self.mujoco_objects, f"Object '{obj.name}' already has sampler associated with it!"
			self.mujoco_objects.append(obj)
		# Make sure sampler_name exists
		assert sampler_name in self.samplers.keys(), "Invalid sub-sampler specified, valid options are: {}, " \
													 "requested: {}".format(self.samplers.keys(), sampler_name)
		# Add the mujoco objects to the requested sub-sampler
		self.samplers[sampler_name].add_objects(mujoco_objects)

	def reset(self):
		"""
		Resets this sampler. In addition to base method, iterates over all sub-samplers and resets them
		"""
		# print("reset in SequentialCompositeSampler")
		super().reset()
		for sampler in self.samplers.values():
			sampler.reset()

	def sample(self, fixtures=None, reference=None, on_top=True):
		"""
		Sample from each placement initializer sequentially, in the order
		that they were appended.

		Args:
			fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
				obstacles that should not be in contact with newly sampled objects. Used to make sure newly
				generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

			reference (str or 3-tuple or None): if provided, sample relative placement. This will override each
				sampler's @reference argument if not already specified. Can either be a string, which
				corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
				relative to this sampler's `'reference_pos'` value.

			on_top (bool): if True, sample placement on top of the reference object. This will override each
				sampler's @on_top argument if not already specified. This corresponds to a sampled
				z-offset of the current sampled object's bottom_offset + the reference object's top_offset
				(if specified)

		Return:
			dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
				placements specified in @fixtures. Note quat is in (w,x,y,z) form

		Raises:
			RandomizationError: [Cannot place all objects]
		"""
		# Standardize inputs
		# print("sample in SequentialCompositeSampler")
		placed_objects = {} if fixtures is None else copy(fixtures)

		# Iterate through all samplers to sample
		for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):
			# Pre-process sampler args
			if s_args is None:
				s_args = {}
			for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
				if arg_name not in s_args:
					s_args[arg_name] = arg
			# Run sampler
			new_placements = sampler.sample(fixtures=placed_objects, **s_args)
			# Update placements
			placed_objects.update(new_placements)
		# print("sample in sequential")
		# print("placed_objects in sq", placed_objects)
		return placed_objects

            rgba = [0,0,0,1],
            friction=[1,1,5],

        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.phone)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.phone,
                x_range=[0.428, 0.728],
                y_range=[-0.3],
                rotation=[1,0,0,0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.00,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.phone,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.phone_body_id = self.sim.model.body_name2id(self.phone.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        # ! TODO ADD THE BELOW LINE IN THE MAIN FILE 
        # self.sim.data.set_joint_qvel('iphone_joint0', [0, -0.2, 0, 0, 0, 0])
        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"


            # phone-related observables
            @sensor(modality=modality)
            def phone_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.phone_body_id])

            @sensor(modality=modality)
            def phone_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.phone_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_phone_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["phone_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "phone_pos" in obs_cache else np.zeros(3)

            sensors = [phone_pos, phone_quat, gripper_to_phone_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # print(f"lift.py -> object_placements {object_placements['iphone'][0]}")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))


    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.phone)

    def _check_success(self):
        """
        Check if phone has been lifted.

        Returns:
            bool: True if phone has been lifted
        """
        phone_height = self.sim.data.body_xpos[self.phone_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # phone is higher than the table top above a margin
        return phone_height > table_height + 0.04

class PickiPhone(Lift):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)