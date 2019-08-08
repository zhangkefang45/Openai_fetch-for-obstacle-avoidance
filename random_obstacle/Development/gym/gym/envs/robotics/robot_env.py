import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class   RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_object(self, name):
        return self.sim.data.get_geom_xpos(name)

    def geom_name2id(self, name):
        pos = self.get_object(name)
        for p in range(len(self.sim.data.geom_xpos)):
            dis = np.sqrt(np.sum(np.square(self.sim.data.geom_xpos - pos)))
            if dis<1e-5:
                return p

    def get_object_range(self, name):
        sim_state = self.sim.get_state()
        x = self.sim.model.get_joint_qpos_addr(name + ":x")
        y = self.sim.model.get_joint_qpos_addr(name + ":y")
        z = self.sim.model.get_joint_qpos_addr(name + ":z")
        x_range = [sim_state.qpos[x]-0.05, sim_state.qpos[x]+0.25]
        y_range = [sim_state.qpos[y]-0.4, sim_state.qpos[y]+0.4]
        z_range = [sim_state.qpos[z]+0.25, sim_state.qpos[z]+0.5]
        return x_range, y_range, z_range

    def random_set_object(self, name, xyz_range):
        sim_state = self.sim.get_state()
        x_joint_i = self.sim.model.get_joint_qpos_addr(name + ":x")
        y_joint_i = self.sim.model.get_joint_qpos_addr(name + ":y")
        z_joint_i = self.sim.model.get_joint_qpos_addr(name + ":z")
        sim_state.qpos[x_joint_i] = np.random.uniform(xyz_range[0][0], xyz_range[0][1])
        sim_state.qpos[y_joint_i] = np.random.uniform(xyz_range[1][0], xyz_range[1][1])
        sim_state.qpos[z_joint_i] = np.random.uniform(xyz_range[2][0], xyz_range[2][1])
        self.sim.set_state(sim_state)
        self.sim.forward()

    def set_object(self, name, new_pos):
        # sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # site_id = self.sim.model.site_name2id(name)
        # self.sim.model.site_pos[site_id] = new_pos
        # self.sim.forward()
        sim_state = self.sim.get_state()
        x_joint_i = self.sim.model.get_joint_qpos_addr(name+":x")
        y_joint_i = self.sim.model.get_joint_qpos_addr(name+":y")
        z_joint_i = self.sim.model.get_joint_qpos_addr(name+":z")
        sim_state.qpos[x_joint_i] += new_pos[0]
        sim_state.qpos[y_joint_i] += new_pos[1]
        sim_state.qpos[z_joint_i] += new_pos[2]
        self.sim.set_state(sim_state)
        # id = self.geom_name2id(name)
        # self.sim.data.geom_xpos[id] = new_pos
        # print(self.get_object(name))
        self.sim.forward()

    def test_step(self, joints):
        self._set_joint(joints)
        # self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def test_get_joint(self):
        return self._get_joint()

    def test_set_joint(self, joints):
        self._to_joints(joints)
        # self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def fix_target(self, goal):
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = goal
        obs = self._get_obs()
        return obs

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)
            # original image is upside-down, so flip it
            return data
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
