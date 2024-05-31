#Why does a pimp need a surveillance center? (C) MIT licensed joshuah rainstar 2024 
#SLICKBACK is a refined adam/SGD optimizer that attempts to find and follow the optimal descent rate.
#it does this through not being stupid
#is not issued with any guarantees to be useful for ur purposes.
#"simple" is for purely linear networks without batching or activation functions that you usually just do param.data -= param.grad
#try it! if your first epoch doesn't beat your conventional approach you probably want simple turned off.
#I  recommend before your first layer in any 2d object recognition network you do x = torch.fft.fftshift(torch.fft.ifft2(x).real) + x
#https://www.youtube.com/watch?v=47jKGHZfplY

class Slickback:
    #https://github.com/falseywinchnet/AI_STUFF
   def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, simple=False):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.simple = simple
        if not simple:
            self.m = [torch.zeros_like(p) for p in self.params]
            self.v = [torch.zeros_like(p) for p in self.params]
            self.t = 0
        self.grad_history = {p: None for p in self.params}
        self.max_shifts = {p: None for p in self.params}

    def step(self):
        if self.simple:
            self.simple_step()
        else:
            self.adam_step()

        def simple_step(self):
        for i, param in enumerate(self.params):
                
                if param.grad is not None:
                  stx = torch.std(param.grad).item()
                  while stx < 0.1:
                    stx = stx * 10
                  sty = torch.norm(param.grad).item()
                  if stx>sty:
                    grad = param.grad * (stx **0.1)
                  else:
                    grad = param.grad/(sty)

                  if self.grad_history[param] is not None:
                    sign_change = (self.grad_history[param] * grad).lt(0)
                    if sign_change.any():
                        max_shift = abs(grad) / 2
                        self.max_shifts[param] = max_shift
                    else:
                        if self.max_shifts[param] is not None:
                            grad = torch.clamp(grad, min=-self.max_shifts[param], max=self.max_shifts[param])

                  self.grad_history[param] = grad.clone()
                  param.data -= grad

    def adam_step(self):
        beta1, beta2 = self.betas
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad.data

                if self.grad_history[param] is not None:
                    sign_change = (self.grad_history[param] * grad).lt(0)
                    if sign_change.any():
                        max_shift = abs(grad) / 2
                        self.max_shifts[param] = max_shift
                    else:
                        if self.max_shifts[param] is not None:
                            grad = torch.clamp(grad, min=-self.max_shifts[param], max=self.max_shifts[param])

                self.grad_history[param] = grad.clone()

                self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)
                param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
