from gymnasium import Wrapper

class RewardsShapingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, terminated, truncated, info  = self.env.step(action)
        
        modified_reward = 0
        if terminated and reward == 1:  # Agent reached the goal
            modified_reward = reward + 10  # Add a bonus reward
        if terminated and reward == 0:  # Agent fell into a hole
            modified_reward = reward - 5
        if not terminated:
            modified_reward = -0.04
        return state, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



