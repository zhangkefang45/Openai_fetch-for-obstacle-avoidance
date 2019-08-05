import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None:
            max_episode_steps = env.spec.max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def test_get_joint(self):
        return self.env.test_get_joint()

    def test_step(self, joints):
        observation, reward, done, info = self.env.test_step(joints)
        return observation, reward, done, info

    def test_set_joint(self, joints):
        observation, reward, done, info = self.env.test_set_joint(joints)
        return observation, reward, done, info

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def fix_target(self, goal):
        return self.env.fix_target(goal)

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
