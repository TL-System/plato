
class WallTime(object):
    """
    A global time object distributed to all 
    tasks, nodes and workers in the environment
    """
    def __init__(self):
        self.curr_time = 0.0

    def update_time(self, new_time):
        self.curr_time = new_time

    def increment_time(self, tick):
        self.curr_time += tick

    def reset(self):
        self.curr_time = 0.0
