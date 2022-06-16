
class WallTime(object):
    """
    A global time object distributed to all workers
    """
    def __init__(self):
        self.curr_time = 0.0

    def update(self, new_time):
        self.curr_time = new_time

    def reset(self):
        self.curr_time = 0.0
