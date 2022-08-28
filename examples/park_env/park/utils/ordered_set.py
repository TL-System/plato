from collections import OrderedDict


class OrderedSet(object):
    def __init__(self, contents=()):
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, item):
        return item in self.set

    def __iter__(self):
        return iter(self.set.keys())

    def __len__(self):
        return len(self.set)

    def __reversed__(self):
        return iter(reversed(self.set.keys()))

    def add(self, item):
        self.set[item] = None

    def clear(self):
        self.set.clear()

    def pop(self):
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, item):
        del self.set[item]

    def to_list(self):
        return [k for k in self.set]