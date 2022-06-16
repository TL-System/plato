
class SetWithCount(object):
    """
    allow duplication in set
    """
    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]