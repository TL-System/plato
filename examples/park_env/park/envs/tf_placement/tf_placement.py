from park.param import config
from park.envs.tf_placement.tf_env import TFRuntime
from park.envs.tf_placement_sim.tf_placement_sim import TFPlacementSimEnv

class TFPlacementEnv(TFPlacementSimEnv):
    """
    Assign a placement to each operation group of a
    computational graph of deep-learning models.
    The goal is to minimize runtime of the computational graph. 

    * STATE *
        Directed Graph with node feature being a list of the following:
            (1) Cost: Op group execution time
            (2) Mem: Op group's memory requirement when running
            (3) Curr Placement: device id of the node based on its 
            current placement in the episode
            (4) is_curr_node: Is this the node that is currently being placed
        

    * ACTIONS *
        [0, 1, ..., n-1] where n is the number of devices. The index
        corresponding to the device id.

    * REWARD *
        Improvement in the runtime of the placement because of the current action
    
    * REFERENCE *
        https://arxiv.org/pdf/1706.04972.pdf
    """
    def __init__(self):
        TFPlacementSimEnv.__init__(self)
        self.tf_runtime = TFRuntime(config.pl_graph, self.device_names)

    # takes op-group placement and 
    # returns runtime of the placement in seconds
    def get_rt(self, pl):
        pl = self.ungroup_pl(pl)

        rt = self.tf_runtime.measure(pl)

        return rt / 1e6
