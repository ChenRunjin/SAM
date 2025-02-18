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
    """
    Functional-SAM 优化器：
      - 要求 closure 每次返回一对 (loss, logits)：
          * 第一次调用 closure 得到 (clean_loss, clean_logits)：未扰动状态下的输出，
            用于计算 dL/dlogits（干净梯度信号）。
          * 第二次调用 closure 得到 (perturbed_loss, perturbed_logits)：扰动状态下的输出，
            用于构造扰动分支的计算图并计算 d(perturbed_logits)/dθ。
      - 最终梯度 = (clean 分支的 dL/dlogits) * (扰动分支的 d(logits)/dθ)
    """
    def __init__(self, params, base_optimizer, rho=0.05, precond=False, **kwargs):
        """
        参数：
          params (iterable): 要优化的参数集合。
          base_optimizer (class): 基础优化器，如 torch.optim.SGD 或 Adam。
          rho (float): 扰动大小。
          **kwargs: 传递给基础优化器的其他参数。
        """
        assert rho >= 0.0, "rho 必须非负"
        defaults = dict(rho=rho, precond=precond, **kwargs)
        super().__init__(params, defaults)

        # 初始化基础优化器
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # 保持 self.param_groups 与基础优化器内部一致
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一次 Functional-SAM 更新步骤。
        要求 closure 返回一对 (loss, logits)，例如：
        
            def closure():
                logits = model(x)
                loss = loss_fn(logits, y)
                return loss, logits

        其中：
          - 第一次调用 closure 得到 (clean_loss, clean_logits)：未扰动状态下的结果，
            用于计算 dL/dlogits（干净梯度）。
          - 调用 first_step 后，再次调用 closure 得到 (perturbed_loss, perturbed_logits)
            用于构造扰动状态下的计算图。
        """
        assert closure is not None, "FunctionalSAM 需要 closure 返回 (loss, logits)"
        closure = torch.enable_grad()(closure)

        # 1. 干净分支：在未扰动状态下前向传播，获得 (clean_loss, clean_logits)
        clean_loss, clean_logits = closure()
        # 注意：这里不能 detach，保留计算图以便计算 dL/dlogits

        # 2. 对参数进行扰动（first_step），保存原始参数以便后续恢复
        self.first_step(zero_grad=True)

        # 3. 扰动分支：在扰动状态下再次前向传播，获得 (perturbed_loss, perturbed_logits)
        perturbed_loss, perturbed_logits = closure()

        # 4. 对扰动状态下的 loss 进行反向传播，建立二阶计算图
        perturbed_loss.backward(create_graph=True)

        # 5. 利用干净分支计算 dL/dlogits（未受扰动影响）
        dL_dlogits = torch.autograd.grad(
            clean_loss,       # 干净 loss
            clean_logits,     # 干净 logits
            retain_graph=True,
            allow_unused=True
        )[0]

        # 6. 使用扰动分支返回的 perturbed_logits 来计算 d(perturbed_logits)/dθ
        all_params = [p for group in self.param_groups for p in group["params"] if p.grad is not None]
        grads = torch.autograd.grad(
            perturbed_logits,  # 扰动状态下的 logits
            all_params,        # 针对每个参数求导
            grad_outputs=dL_dlogits,  # 使用干净分支计算得到的 dL/dlogits
            retain_graph=True
        )

        # 7. 恢复原始参数（未扰动前的参数）
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])

        # 8. 将计算得到的梯度赋值给对应参数
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad = grads[idx]
                idx += 1

        # 9. 使用基础优化器进行参数更新，并清零梯度
        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        根据当前梯度，对每个参数进行扰动：
            perturb = scale * p.grad，其中 scale = rho / (||grad|| + 1e-12)
        同时保存原始参数以便后续恢复。
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 保存原始参数
                self.state[p]["old_p"] = p.clone().detach()
                # 计算扰动，并进行 in-place 更新
                e_w = p.grad.detach() * scale.to(p.device)
                if group["precond"] and "exp_avg_sq" in self.base_optimizer.state[p]:
                    beta2 = group["betas"][1]
                    bias_correction = 1.0 - beta2 ** self.base_optimizer.state[p]["step"]
                    e_w /= (self.base_optimizer.state[p]["exp_avg_sq"].sqrt() + 1e-12) / bias_correction
                p.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self):
        """
        计算所有梯度的全局 2 范数。
        """
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
        """
        确保基础优化器加载正确的状态。
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

