# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------
"""
Nvidia Source Code License-NC

1. Definitions

“Licensor” means any person or entity that distributes its Work.

“Software” means the original work of authorship made available under this License.
“Work” means the Software and any additions to or derivative works of the Software that are made available under
this License.

“Nvidia Processors” means any central processing unit (CPU), graphics processing unit (GPU), field-programmable gate
array (FPGA), application-specific integrated circuit (ASIC) or any combination thereof designed, made, sold, or
provided by Nvidia or its affiliates.

The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S.
copyright law; provided, however, that for the purposes of this License, derivative works shall not include works that
remain separable from, or merely link (or bind by name) to the interfaces of, the Work.

Works, including the Software, are “made available” under this License by including in or with the Work either
(a) a copyright notice referencing the applicability of this License to the Work, or (b) a copy of this License.

2. License Grants

2.1 Copyright Grant. Subject to the terms and conditions of this License, each Licensor grants to you a perpetual,
worldwide, non-exclusive, royalty-free, copyright license to reproduce, prepare derivative works of, publicly display,
publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.

3. Limitations

3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this License, (b) you include
a complete copy of this License with your distribution, and (c) you retain without modification any copyright, patent,
trademark, or attribution notices that are present in the Work.

3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and
distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation
in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to
Your Terms. Notwithstanding Your Terms, this License (including the redistribution requirements in Section 3.1) will
continue to apply to the Work itself.

3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially.
The Work or derivative works thereof may be used or intended for use by Nvidia or its affiliates commercially or
non-commercially.  As used herein, “non-commercially” means for research or evaluation purposes only.

3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim,
cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then
your rights under this License from such Licensor (including the grants in Sections 2.1 and 2.2) will terminate
immediately.

3.5 Trademarks. This License does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or
trademarks, except as necessary to reproduce the notices described in this License.

3.6 Termination. If you violate any term of this License, then your rights under this License (including the grants
in Sections 2.1 and 2.2) will terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
WARRANTIES OR CONDITIONS OF M ERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT.
YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE.

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE),
CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
(INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR
MALFUNCTION, OR ANY OTHER COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY
OF SUCH DAMAGES.
"""

# :>
import torch


class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


# The long copyright note might be excessive for these few lines :>
