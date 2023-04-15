# Attack implementations

This module implements all attacks. A new attack should inherit from `base_attack._BaseAttacker` or at least follow the interface outlined there,
which requires that a `reconstruct` method is implemented which takes as input `server_payload` and  `shared_data` which are both lists of payloads and data outputs from `server.distribute_payload` and `user.compute_local_updates`. The attack should return a dictionary that contains the entries `data` and `labels`. Both should be PyTorch tensors that (possibly, depending on the success of the attack) approximate the immediate input to the user model.

Any new optimization-based attack can likely inherit a lot of functionality already present in `optimization_based_attack.OptimizationBasedAttacker`.

Implementing a new regularizer or objective requires no change to the main attacks, only another entry in the  `auxiliaries/regularizers.regularizer_lookup` or `auxiliaries/objectives.objective_lookup` interface respectively.
