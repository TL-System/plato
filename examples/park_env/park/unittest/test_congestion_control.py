import park

class RandomAgent(object):
    def __init__(self, state_space, action_space, *args, **kwargs):
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self, obs, prev_reward, prev_done, prev_info):
        act = self.action_space.sample()
        # implement real action logic here
        return act

def main():
    env = park.make('congestion_control')
    env.run(RandomAgent, ())
