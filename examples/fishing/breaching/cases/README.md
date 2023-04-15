# Cases

This module implements the core functionality for each use case. `users.py` implements the user-side protocol for different types of users (`single_gradient`, multiple `local_updates` and aggregations of such users as `multiuser_aggregate`). `servers.py` implements a range of server threat models from `honest-but-curious` servers to `malicious-model` and `malicious-parameters` variants.

If you are looking specifically for the implementation of the modification necessary for "Robbing-the-Fed", take a look at `malicious_modifications/imprint.py`. If you are looking for "Decepticons", look at `malicious_modifications/analytic_transformer_utils.py`.
