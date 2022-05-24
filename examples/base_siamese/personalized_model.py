""" The personalized model for the client. 

    - Linear evaluation. 
        Based on the learnt representation, each client creates linear network to
      perform its own task, such as image classification.
"""

import torch
import torch.nn as nn
