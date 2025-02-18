import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, precond=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, precond=precond, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None: continue
        #         if group["precond"] and "exp_avg_sq" in self.base_optimizer.state[p]:
        #             p.grad.div_(self.base_optimizer.state[p]["exp_avg_sq"].sqrt() + 1e-12)
                
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                if group["precond"] and "exp_avg_sq" in self.base_optimizer.state[p]:
                    beta2 = group["betas"][1]
                    bias_correction = 1.0 - beta2 ** self.base_optimizer.state[p]["step"]
                    e_w /= (self.base_optimizer.state[p]["exp_avg_sq"].sqrt() + 1e-12)/bias_correction
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups




class FunctionalSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, precond=False, **kwargs):
        assert rho >= 0.0, "rho must be positive"
        defaults = dict(rho=rho, precond=precond, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FunctionalSAM need closure return (loss, logits)"
        closure = torch.enable_grad()(closure)

        clean_loss, clean_logits = closure()
        
        self.first_step(zero_grad=True)

        perturbed_loss, perturbed_logits = closure()

        perturbed_loss.backward(create_graph=True)

        dL_dlogits = torch.autograd.grad(
            clean_loss,       
            clean_logits,     
            retain_graph=True,
            allow_unused=True
        )[0]

        all_params = [p for group in self.param_groups for p in group["params"] if p.grad is not None]
        grads = torch.autograd.grad(
            perturbed_logits, 
            all_params,       
            grad_outputs=dL_dlogits,  
            retain_graph=True
        )

  
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad = grads[idx]
                idx += 1

        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.clone().detach()
                e_w = p.grad.detach() * scale.to(p.device)
                if group["precond"] and "exp_avg_sq" in self.base_optimizer.state[p]:
                    beta2 = group["betas"][1]
                    bias_correction = 1.0 - beta2 ** self.base_optimizer.state[p]["step"]
                    e_w /= (self.base_optimizer.state[p]["exp_avg_sq"].sqrt() + 1e-12) / bias_correction
                p.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(shared_device))
        if len(norms) == 0:
            return torch.tensor(0., device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

