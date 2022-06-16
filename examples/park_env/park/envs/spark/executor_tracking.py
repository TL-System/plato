from collections import OrderedDict


class ExecutorMap(object):

    """
    This class keeps track of the executor flow in spark, such that
    the Master and DAGScheduler can coordinate the executor assignments
    """

    def __init__(self):

        # dynamically add {app_id -> num_executors} to OrderedDict
        # initiate moving of executors, so that spark master knows
        # which app to move the executors to when it kills an executor
        self.executor_flow = OrderedDict()

    def add_executor_flow(self, app_id, num_executors):
        # record to move num_executors to app_id
        if app_id not in self.executor_flow:
            self.executor_flow[app_id] = 0
        self.executor_flow[app_id] += num_executors

    def pop_executor_flow(self, num_available_executors):
        # move one executor in the flow
        # decrement the target amount by number of executors moved

        if len(self.executor_flow) == 0:
            return "void", 0

        move_executor_to_app = next(iter(self.executor_flow))
        num_executors_moved = min(
            num_available_executors,
            self.executor_flow[move_executor_to_app])

        self.executor_flow[move_executor_to_app] -= num_executors_moved

        if self.executor_flow[move_executor_to_app] == 0:
            del self.executor_flow[move_executor_to_app]

        return move_executor_to_app, int(num_executors_moved)

    def remove_app(self, app_id):

        if app_id in self.executor_flow:
            del self.executor_flow[app_id]
