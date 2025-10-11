import torch


class FedMosOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, a=1.0, mu=0.0):
        defaults = dict(lr=lr, a=a, mu=mu)
        super(FedMosOptimizer, self).__init__(params, defaults)

    def clone_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                gt = p.grad.data
                if gt is None:
                    continue
                self.state[p]["gt_prev"] = gt.clone().detach()

    def get_grad(self):
        grad = []
        for group in self.param_groups:
            for p in group["params"]:
                gt = p.grad.data
                if gt is None:
                    continue
                grad += [gt.clone().detach().cpu().numpy()]
        return grad

    def update_momentum(self):
        for group in self.param_groups:
            for p in group["params"]:
                gt = p.grad.data  # grad
                if gt is None:
                    continue
                a = group["a"]
                state = self.state[p]
                if len(state) == 0:
                    # State initialization
                    state["gt_prev"] = torch.zeros_like(p.data)
                    state["dt"] = gt.clone()
                    continue

                    # state['gt_prev'] = torch.zeros_like(p.data)
                    # state['dt'] = gt.clone()

                gt_prev = state["gt_prev"]
                # assert not torch.allclose(gt, gt_prev), 'Please call clone_grad() in the preious step.'
                dt = state["dt"]
                # print(torch.equal(dt-gt_prev, torch.zeros_like(dt-gt_prev)))
                # print(dt-gt_prev)
                state["dt"] = gt + (1 - a) * (dt - gt_prev)
                state["gt_prev"] = gt.clone().detach()
                # state['gt_prev'] = None

    def step(self, local_net):
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            for p, local_p in zip(group["params"], local_net.parameters()):
                state = self.state[p]
                if len(state) == 0:
                    raise Exception("Please call update_momentum() first.")

                lr, mu = group["lr"], group["mu"]
                dt = state["dt"]
                prox = p.data - local_p.data
                p.data.add_(dt, alpha=-lr)
                p.data.add_(prox, alpha=-mu)
