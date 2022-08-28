__all__ = ['AttrDict', 'flatten', 'nested_update', 'nested_select', 'nested_setdefault', 'ordered_flatten']


class AttrDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(item)
        return self[item]

    def __setattr__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)

    @classmethod
    def nested_attr(cls, data: dict):
        results = {k: cls.nested_attr(v) if isinstance(v, dict) else v for k, v in data.items()}
        return AttrDict(**results)


def flatten(d):
    if isinstance(d, dict):
        results = []
        for i in d.values():
            if isinstance(i, dict):
                results.extend(flatten(i))
            else:
                results.append(i)
        return results
    else:
        return d


def ordered_flatten(d):
    if isinstance(d, dict):
        results = []
        for key in sorted(d.keys()):
            if isinstance(d[key], dict):
                results.extend(ordered_flatten(d[key]))
            else:
                results.append(d[key])
        return results
    else:
        return d


def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def nested_setdefault(d, u, default=None):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = nested_setdefault(d.get(k, {}), v, default)
        else:
            if default is not None:
                d.setdefault(k, default)
            else:
                d.setdefault(k, v[k])
    return d


def nested_select(d, v, default_selected=True):
    """
    Nestedly select part of the object d with indicator v. If d is a dictionary, it will continue to select the child
    values. The function will return the selected parts as well as the dropped parts.
    :param d: The dictionary to be selected
    :param v: The indicator showing which part should be selected
    :param default_selected: Specify whether an entry is selected by default
    :return: A tuple of two elements, the selected part and the dropped part
    Examples:
    >>> person = {'profile': {'name': 'john', 'age': 16, 'weight': 85}, \
                  'relatives': {'father': 'bob', 'mother': 'alice'}}
    >>> nested_select(person, True)
    ({'profile': {'name': 'john', 'age': 16, 'weight': 85}, 'relatives': {'father': 'bob', 'mother': 'alice'}}, {})
    >>> nested_select(person, {'profile': False})
    ({'relatives': {'father': 'bob', 'mother': 'alice'}}, {'profile': {'name': 'john', 'age': 16, 'weight': 85}})
    >>> nested_select(person, {'profile': {'name': False}, 'relatives': {'mother': False}})
    ({'profile': {'age': 16, 'weight': 85}, 'relatives': {'father': 'bob'}},
     {'profile': {'name': 'john'}, 'relatives': {'mother': 'alice'}})
    """

    if isinstance(v, dict):
        assert isinstance(d, dict)

        choosed = d.__class__()
        dropped = d.__class__()

        for k in d:
            if k not in v:
                if default_selected:
                    choosed.setdefault(k, d[k])
                else:
                    dropped.setdefault(k, d[k])
                continue

            if isinstance(v[k], dict):
                assert isinstance(d[k], dict)

                child_choosed, child_dropped = nested_select(d[k], v[k])
                if child_choosed:
                    choosed.setdefault(k, child_choosed)
                if child_dropped:
                    dropped.setdefault(k, child_dropped)
            else:
                if v[k]:
                    choosed.setdefault(k, d[k])
                else:
                    dropped.setdefault(k, d[k])

        return choosed, dropped
    else:
        other = d.__class__() if isinstance(d, dict) else None
        return (d, other) if v else (other, d)
