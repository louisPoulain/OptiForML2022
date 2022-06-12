import torch
import math

class AdaSGD(torch.optim.Optimizer):
    # Mix between AdaHessian and SGD
    """
    Arguments:
    params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
    lr (float, optional) -- learning rate (default: 0.1)
    betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
    eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
    weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
    hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
    update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
    n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    momentum (float, optional) -- momentum for th sgd part (default: 0.95)
    ada_w=0.5 
    sgd_w=0.5
    """

    def __init__(self,
                 params, lr=0.1, weight_decay=0.0,
                 betas=(0.9, 0.999), eps=1e-8, dampening = 0,
                 momentum=0.95, nesterov=False, ada_w=0.5, sgd_w=0.5,
                 hessian_power=1.0, update_each=1, n_samples=1, 
                 average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            betas=betas, eps=eps, dampening=dampening,
            momentum=momentum, nesterov = nesterov, ada_w=ada_w, sgd_w=sgd_w,
            hessian_power = hessian_power, 
        )

        super(AdaSGD, self).__init__(params, defaults)

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0
    
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            #group.setdefault('maximize', False)

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.zero_hessian()
        self.set_hessian()
        for group in self.param_groups:
            for p in group['params']:
                
                if p.grad is None or p.hess is None:
                    continue
                grad = p.grad
                hess = p.hess
                #if grad.is_sparse:
                #    raise RuntimeError('AdaSGD does not support sparse gradients')

                d_p_adaH, step_size = self.ada_step(group, grad, hess, p)

                d_p_sgd = self.sgd_step(grad, group, p)

                merged_d_p = group['sgd_w'] * d_p_sgd + group['ada_w'] * d_p_adaH
                #print(f'[{d_p_adaH}, {d_p_sgd}, {merged_d_p}],')
                merged_lr = group['sgd_w'] * group['lr'] + group['ada_w'] * step_size

                p.add_(merged_d_p, alpha=-merged_lr)

        return loss

    @torch.no_grad()
    def ada_step(self, grad, hess, group, p): # p = w_k, trying to get w_{k+1}
        # diag(H) in paper given by set_hessian so here this is just hess (since in step we already used set_hessina for all params)
        d_p = grad
        if self.average_conv_kernel and p.dim() == 4:
            p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

        # Perform correct stepweight decay as in AdamW
        # w_{k+1} = (1 - lr * weight_decay) w_k in AdamW
        #p.mul_(1 - group['lr'] * group['weight_decay'])
        state = self.state[p]

        # State initialization
        if len(state) == 1:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
            state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

        exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq'] #exp_avg = m_{k}, exp_hessian = v_{k}, first and second moments
        beta1, beta2 = group['betas']
        state['step'] += 1

        # Decay the first and second moment running average coefficient
            # exp_avg = beta1 * exp_avg + (1 - beta1) * grad, see eq (12) adaHessian paper, first line
            # m_{k+1} = beta1 * m_k + (1 - beta1) * grad (paper is so badly explained)
        exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
           
            # same, very badly explained, better to refer to adam algo
            # v_{k+1} = beta2 * v_k + (1-beta2) * D ** 2 (slight chnage wrt Adam, chnage grad ** 2 into hessian ** 2)
            # addcmul: input = input + hess * hess * value
            # exp_hess = beta2 * exp_hess + (1 - beta2) * hess ** 2 (see eq (12) in adahessian paper)
        exp_hessian_diag_sq.mul_(beta2).addcmul_(hess, hess, value=1 - beta2) 

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        k = group['hessian_power']
        denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps']) # cf expression for v_{k+1}, eq (12)

        # make update
        # w_{k+1} = w_k - lr * (m_k / (1 - beta1 ** k)) / (v_k / (1 - beta2 ** k))
        # with adamW update: w_{k+1} = w_k - (lr * exp_avg / bias_correction1 / denom - weight_decay * w_k)
        step_size = group['lr'] / bias_correction1
        d_p = p - (step_size * exp_avg / denom - group['weight_decay'] * p)
        
        return d_p, step_size

    def sgd_step(self, grad, group, p):
        d_p = grad
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        if weight_decay != 0:
            d_p = d_p.add(p, alpha=weight_decay) # d_p = d_p + alpha * p
        if momentum != 0:
            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        return d_p
