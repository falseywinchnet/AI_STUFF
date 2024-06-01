# (C) MIT licensed joshuah rainstar 2024 
#in truth i am an idiot and colab is cursed because sometimes optimizers simply have no effect
#and whenever i tried to run adam it didnt work but this did
#so i misunderstood the benefit of what i had done
#this was just adam with extra steps
#i have left the simple because it is still novel and i managed to do osmething useful with it

class Fast_SGD:
    #https://github.com/falseywinchnet/AI_STUFF
    def __init__(self, params):

        def step(self):
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

                  param.data -= grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
