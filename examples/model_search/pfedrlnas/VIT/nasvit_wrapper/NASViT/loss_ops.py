# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

# def alpha_divergence(q_logits, p_logits, alpha, reduction='none'):
#    q_prob = torch.nn.functional.softmax(q_logits, dim=1)
#    p_prob = torch.nn.functional.softmax(p_logits, dim=1)
#
#    if isinstance(alpha, float) and abs(alpha) < 1e-3:
#        lndiff = q_prob.log() - p_prob.log()
#        lndiff.clamp_(-20, 20)
#        loss = torch.sum(q_prob * lndiff, dim=1) # KL(q||p)
#    elif isinstance(alpha, float) and abs(alpha - 1.0) < 1e-3:
#        loss = torch.sum(p_prob * (p_prob.log() - q_prob.log()), dim=1) # KL(p||q)
#    else:
#        if isinstance(alpha, float):
#            iw_ratio = torch.pow(p_prob/q_prob, alpha)
#        else:
#            iw_ratio = torch.pow(p_prob/q_prob, alpha.unsqueeze(-1))
#        iw_ratio = iw_ratio.clamp(0, 10)
#        loss = 1.0 / (alpha*(alpha-1.0)) * ((iw_ratio * q_prob).sum(1) - 1.0) # D_a(p||q)
#
#    if reduction == 'mean':
#        return loss.mean()
#    elif reduction == 'sum':
#        return loss.sum()
#    return loss


def alpha_divergence(q_logits, p_logits, alpha, reduction="none", iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1)
    p_prob = torch.nn.functional.softmax(p_logits, dim=1)

    if abs(alpha) < 1e-3:
        lndiff = q_prob.log() - p_prob.log()
        lndiff.clamp_(-iw_clip, iw_clip)
        loss = torch.sum(q_prob * lndiff, dim=1)  # KL(q||p)
    elif abs(alpha - 1.0) < 1e-3:
        loss = torch.sum(p_prob * (p_prob.log() - q_prob.log()), dim=1)  # KL(p||q)
    else:
        iw_ratio = torch.pow(p_prob / q_prob, alpha)
        iw_ratio = iw_ratio.clamp(0, iw_clip)
        loss = (
            1.0 / (alpha * (alpha - 1.0)) * ((iw_ratio * q_prob).sum(1) - 1.0)
        )  # D_a(p||q)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def f_min_divergence(
    q_logits, p_logits, alpha_left, alpha_right, iw_clip=1e3, p_normalize=False
):
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    if p_normalize:
        p_prob = p_logits.detach()
    else:
        p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()

    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=1)
    importance_ratio = p_prob / q_prob

    iw_alpha_left = torch.pow(importance_ratio, alpha_left)
    iw_alpha_left = iw_alpha_left.clamp(0, iw_clip)
    f_left = iw_alpha_left / alpha_left / (alpha_left - 1.0)
    f_base_left = 1.0 / alpha_left / (alpha_left - 1.0)
    rho_f_left = iw_alpha_left / alpha_left + f_base_left

    iw_alpha_right = torch.pow(importance_ratio, alpha_right)
    iw_alpha_right = iw_alpha_right.clamp(0, iw_clip)
    f_right = iw_alpha_left / alpha_right / (alpha_right - 1.0)
    f_base_right = 1.0 / alpha_right / (alpha_right - 1.0)
    rho_f_right = iw_alpha_right / alpha_right + f_base_right

    ind = torch.gt(iw_alpha_left, iw_alpha_right)

    loss = torch.sum(
        q_prob * ((f_left - f_base_left) * ind + (f_right - f_base_right) * (1 - ind)),
        dim=1,
    )
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3, p_normalize=False):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    if p_normalize:
        p_prob = p_logits.detach()
    else:
        p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
    """
    # smooth p_prob
    local_label_smooth = .1
    p_prob = p_prob + torch.ones_like(p_prob) * local_label_smooth / 1000. 
    p_prob /= (local_label_smooth + 1)
    """
    q_log_prob = torch.nn.functional.log_softmax(
        q_logits, dim=1
    )  # gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification"""

    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss.mean()


class FdivTopKLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    """

    def forward(self, output, target, T=1.0):
        output, target = output / T, target / T
        output_prob = torch.nn.functional.softmax(output, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)

        target_prob = torch.nn.functional.softmax(target, dim=1)
        # density ratios
        density_ratio = target_prob / output_prob
        # _, indices = torch.topk(density_ratio, k, dim=1, largest=True)
        # one_hot_w = torch.zeros_like(target).scatter(1, indices, 1)
        one_hot_w = torch.ge(density_ratio, 1.0).float()

        ##probablity
        # _, indices = torch.topk(target_prob, k, dim=1, largest=True)
        # one_hot_p = torch.zeros_like(target).scatter(1, indices, 1)

        # one_hot = one_hot_w * one_hot_p
        one_hot = one_hot_w.detach()
        # loss = -torch.sum( target_prob * one_hot * output_log_prob, dim=1)
        loss = -torch.sum(target_prob.detach() * one_hot * output_log_prob, dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HardThresLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    """

    def forward(self, output, target, eps=0.1):
        output_prob = torch.nn.functional.softmax(output, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)

        target_prob = torch.nn.functional.softmax(target, dim=1)
        one_hot = torch.ge(target_prob, eps).float()
        n_class = output.size(1)
        noise_labels = torch.sum(target_prob * (1.0 - one_hot), 1, keepdim=True) / (
            n_class - torch.sum(one_hot, 1, keepdim=True)
        )
        target_prob = one_hot * target_prob + (1.0 - one_hot) * noise_labels

        loss = -torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TopkLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    """

    def forward(self, output, target, k=5):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)

        target_prob = torch.nn.functional.softmax(target, dim=1)
        topk_vals, topk_idxs = torch.topk(target_prob, k, dim=1, largest=True)
        one_hot = torch.zeros_like(target).scatter(1, topk_idxs, 1)  # topk, one hot
        n_class = output.size(1)
        noise_labels = torch.sum(target_prob * (1.0 - one_hot), 1, keepdim=True) / (
            n_class - k
        )
        target_prob = one_hot * target_prob + (1.0 - one_hot) * noise_labels

        loss = -torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
#    """ inplace distillation for image classification
#            output: output logits of the student network
#            target: output logits of the teacher network
#            grad_theta = -E_q[ rho_f(p/q) grad_theta \log q]
#    """
#    def forward(self, output, target, eps=1e-5, scale=.1):
#        output_prob = torch.nn.functional.softmax(output, dim=1)
#        target_prob = torch.nn.functional.softmax(target, dim=1)
#        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
#        target_log_prob = torch.nn.functional.log_softmax(target, dim=1)
#
#        rho_f = target_log_prob - output_log_prob - 1
#        rho_var = torch.var(rho_f, dim=1, keepdim=True)
#        rho_f = rho_f / torch.sqrt(rho_var + eps) * scale
#
#        loss = -torch.sum(output_prob.detach() * rho_f.detach() * output_log_prob, dim=1)
#        if self.reduction == 'mean':
#            return loss.mean()
#        elif self.reduction == 'sum':
#            return loss.sum()
#        return loss
#


class KLLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    KL(p||q) = Ep \log p - \Ep log q
    """

    def forward(self, output, soft_logits, target=None, temperature=1.0, alpha=0.9):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            n_class = output.size(1)
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            target = target.unsqueeze(1)
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, output_log_prob).squeeze()
            loss = alpha * temperature * temperature * kd_loss + (1.0 - alpha) * ce_loss
        else:
            loss = kd_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ReverseKLLossSoft(torch.nn.modules.loss._Loss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    KL(q||p) = Eq(\log q - \log p)
    """

    def forward(self, output, target, T=1.0):
        output, target = output / T, target / T
        output_prob = torch.nn.functional.softmax(output, dim=1)
        target_log_prob = torch.nn.functional.log_softmax(target, dim=1)

        loss = torch.sum(output_prob * (output_prob.log() - target_log_prob), dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    def __init__(self, alpha_min, alpha_max, iw_clip=5):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(
        self,
        output,
        target,
        alpha_min=None,
        alpha_max=None,
        mode_seeking_weight=None,
        p_normalize=False,
        reduction=True,
    ):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        # loss_left = alpha_divergence(output, target, alpha_min, iw_clip=self.iw_clip)
        # loss_right = alpha_divergence(output, target, alpha_max, iw_clip=self.iw_clip)
        # loss = torch.max(loss_left, loss_right)
        if mode_seeking_weight is None:
            loss_left, grad_loss_left = f_divergence(
                output, target, alpha_min, iw_clip=self.iw_clip, p_normalize=p_normalize
            )
            loss_right, grad_loss_right = f_divergence(
                output, target, alpha_max, iw_clip=self.iw_clip, p_normalize=p_normalize
            )

            # change max -> min
            ind = torch.gt(loss_left, loss_right).float()
            loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        else:
            alpha = alpha_min * mode_seeking_weight + alpha_max * (
                1.0 - mode_seeking_weight
            )
            _, loss = f_divergence(output, target, alpha, iw_clip=self.iw_clip)
            # loss = mode_seeking_weight * grad_loss_left + (1.0 - mode_seeking_weight) * grad_loss_right

        if not reduction:
            return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # def forward(self, output, target, interpolation=1.0):
    #    loss_left = alpha_divergence(output, target, self.alpha_min)
    #    loss_right = alpha_divergence(output, target, self.alpha_max)
    #
    #    loss_min = torch.min(loss_left, loss_right)
    #    loss_max = torch.max(loss_left, loss_right)
    #    loss = (1.0 - interpolation)*loss_min + interpolation*loss_max
    #
    #    if self.reduction == 'mean':
    #        return loss.mean()
    #    elif self.reduction == 'sum':
    #        return loss.sum()
    #    return loss


class AlphaDivergenceLossSoft(torch.nn.modules.loss._Loss):
    """alpha divergence
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    D_a(q||p) = 1/(a(a-1)) * E_q[p^a q^-a -1] = 1/(a(a-1)) (\sum p^a q^1-a - 1)
    """

    def forward(self, output, target, alpha):
        loss = alpha_divergence(output, target, alpha, reduction="none")
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class EntropyAlphaDivergence(torch.nn.modules.loss._Loss):
    def forward(self, output, target):
        prob = torch.nn.functional.softmax(target, dim=1)
        ent = -torch.sum(prob * prob.log(), dim=1)
        # alpha = (ent - torch.min(ent)) / (torch.max(ent) - torch.min(ent))
        alpha = 1.0 - torch.max(prob, 1).values
        # alpha = torch.max(prob, 1).values
        loss = alpha_divergence(output, target, alpha, reduction="none")

        thr = torch.gt(torch.max(prob, 1).values, 0.8).float()
        forward_kl = alpha_divergence(output, target, 0.0, reduction="none")
        reverse_kl = alpha_divergence(output, target, 1.0, reduction="none")
        loss = thr * forward_kl + (1.0 - thr) * reverse_kl

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


"""
    idea: cross entropy loss doesn't satisfy triangle inequality
          we might want to use a symmetric divergence
"""


class JSDLossSoft(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(JSDLossSoft, self).__init__()

    # {{\rm {JSD}}}(P\parallel Q)={\frac  {1}{2}}D(P\parallel M)+{\frac  {1}{2}}D(Q\parallel M)
    def forward(self, output, target):
        output_prob = torch.nn.functional.softmax(output, dim=1)
        target_prob = torch.nn.functional.softmax(target, dim=1)

        M = (output_prob + target_prob) / 2.0
        # student network
        kl_qm = output_prob * (output_prob.log() - M.log())
        kl_pm = target_prob * (target_prob.log() - M.log())
        loss = torch.sum(0.5 * (kl_qm + kl_pm), dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class JSDLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.1):
        super(JSDLossSmooth, self).__init__()
        self.eps = label_smoothing

    # {{\rm {JSD}}}(P\parallel Q)={\frac  {1}{2}}D(P\parallel M)+{\frac  {1}{2}}D(Q\parallel M)
    def forward(self, output, target):
        output_prob = torch.nn.functional.softmax(output, dim=1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target_prob = one_hot * (1 - self.eps) + self.eps / n_class

        M = (output_prob + target_prob) / 2.0
        # student network
        kl_qm = output_prob * (output_prob.log() - M.log())
        kl_pm = target_prob * (target_prob.log() - M.log())
        loss = torch.sum(0.5 * (kl_qm + kl_pm), dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyLossSmooth, self).__init__()
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output, target, reduction=True):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if not reduction:
            return loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CrossEntropyEma(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyEma, self).__init__()
        self.eps = label_smoothing

    def _forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)
        return loss.squeeze(-2).squeeze(-1)

    def forward(self, output, target, ema_output=None, beta=0.1):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class

        loss_model = self._forward(output, target)
        if ema_output is not None:
            loss_ema = self._forward(ema_output, target)
            loss_model_ema = self._forward(
                output, torch.nn.functional.softmax(ema_output, dim=1)
            )
            indicators = torch.ge(loss_model, loss_ema).float() * beta
            loss = (1.0 - indicators) * loss_model + indicators * loss_model_ema
        else:
            loss = loss_model
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


if __name__ == "__main__":
    output, target = torch.rand(4, 100), torch.rand(4, 100)
    print(output.size(), target.size())
    print(alpha_divergence(output, target, alpha=-0.1))
    print(f_divergence(output, target, alpha=-0.1))
#
#    #soft_target = torch.rand(16, 10)
#    #soft_prob = torch.nn.functional.softmax(soft_target, dim=1)
#    #k = 3
#    #vals, indices = torch.topk(soft_prob, k, largest=True)  # top k largest elements
#    #print(vals)
#    #print(indices)
#    #one_hot = torch.zeros_like(soft_prob).scatter(1, indices, 1)
#    #print(one_hot)
#
#    masks = torch.zeros(16)
#    masks[indices] = 1.
#
#    print(k)
#    print(entropy)
#    print(masks)
#
#    target = torch.rand(6, 10)
#    output = torch.rand(6, 10)
#    criterion = KLLossSoft(reduction='mean')
#    print(criterion(output, target))
#
#    loss = torch.nn.functional.kl_div(
#        torch.nn.functional.log_softmax(output, dim=1),
#        torch.nn.functional.softmax(target, dim=1),
#        reduction='none'
#    )
#    print(loss.sum(1).mean())
#
#    oldc = CrossEntropyLossSoft()
#    print(oldc(output, torch.nn.functional.softmax(target, dim=1)))
#
