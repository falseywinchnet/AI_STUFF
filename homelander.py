# (C) MIT licensed joshuah rainstar 2024 
# doubles convergence rate for some networks(tested on MINST CNN, linear network) but only adds O(N) complexity operations
#is so obvious you're gonna ask yourself how come we aint been doin it already
#the biggest change here is that what we do is if we see a zero crossing(in the gradient!)
#it means that we are oscillating, so every time that happens we clamp the gradient to one-half in subsequent updates.
#we do this recursively to ensure that oscillatory over-steps converge to the target.
#it may be necessary to periodically unset gradhistory and max_shifts, but my testing has not demonstrated a benefit
#use simple=True for non-gated linear networks only, experimental approach
#additional tip for 2d convolve: try  x = torch.fft.fftshift(torch.fft.ifft2(x).real) + x for the first operation

class Homelander:
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
                  if stx >0:
                    while stx < 0.1:
                      stx = stx * 10
                  sty = torch.norm(param.grad).item()
                  if stx>sty:
                    grad = param.grad * (stx **0.1)
                  else:
                    if sty >0:
                        grad = param.grad/(sty)
                    else:
                        grad = param.grad


                  if self.grad_history[param] is not None:
                    sign_change = (self.grad_history[param] * grad).lt(0)
                    if sign_change.any():
                        max_shift = abs(grad) / 2
                        self.max_shifts[param] = max_shift
                        #optional: here, test if maxshifts is not none
                        #if so, make the new maxshifts even smaller
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
