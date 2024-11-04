"""
      Runge-Kutta-Chebyshev Optimizer (RCO) 2024 copyright joshuah rainstar joshuah.rainstar@gmail.com
      licensed under MIT license
"""

from torch.optim import Optimizer
class RCO(Optimizer):
    """
      Runge-Kutta-Chebyshev Optimizer (RCO) - A neural network optimizer that combines 
      4th order Runge-Kutta method and Chebyshev polynomial interpolation with Adam-style 
      momentum.

      This optimizer implements a novel approach to neural network optimization by:
      1. Computing gradients using both RK4 and Chebyshev methods
      2. Averaging the results of both methods
      3. Applying the combined update to the model parameters

      The optimizer uses four evaluation points for both RK4 and Chebyshev methods,
      providing potentially better convergence properties than traditional first-order
      methods.

      Parameters
      ----------
      model : torch.nn.Module
          The neural network model to optimize
      lr : float, optional (default=1e-3)
          Learning rate for the optimizer
      betas : tuple[float, float], optional (default=(0.9, 0.999))
          Coefficients used for computing running averages of gradient and its square
      eps : float, optional (default=1e-8)
          Term added to the denominator to improve numerical stability

      Attributes
      ----------
      model : torch.nn.Module
          The neural network model being optimized
      m : list[torch.Tensor]
          First moment vectors (Adam)
      v : list[torch.Tensor]
          Second moment vectors (Adam)
      t : int
          Number of steps taken

      Methods
      -------
      compute_loss(x, y)
          Computes the MSE loss between model predictions and target values
      step(x, y)
          Performs a single optimization step using the combined RK4-Chebyshev method

      Notes
      -----
      The optimizer implements two main numerical schemes:
      - Runge-Kutta 4th order (RK4) integration with standard coefficients
      - Chebyshev polynomial interpolation using optimized nodes and weights
      
      The final update is computed as an average of both methods, potentially
      providing better stability and convergence properties than either method alone.

      Example
      -------
      >>> model = torch.nn.Linear(10, 1)
      >>> optimizer = RCO(model, lr=1e-3)
      >>> for epoch in range(num_epochs):
      ...     loss = optimizer.step(x_batch, y_batch)
     """
      def __init__(self, model):  # Add clip_value parameter
        self.model = model


    def compute_loss(self, x, y):
        y_pred = self.model(x)
        return torch.mean((y_pred - y)**2)
        
    def step(self, x, y):
        # Initial k1
        loss = self.compute_loss(x, y)
        loss.backward()
        k1 = [p.grad.clone() for p in self.model.parameters()]
        
        # Store original params
        orig_params = [p.data.clone() for p in self.model.parameters()]
        
        # RK4
        for p, k in zip(self.model.parameters(), k1):
            p.data -= k *6
        loss = self.compute_loss(x, y)
        loss.backward()
        k2_rk4 = [p.grad.clone() for p in self.model.parameters()]
        
        # Reset and move to k3 position
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k2_rk4):
            p.data -= k *4
        loss = self.compute_loss(x, y)
        loss.backward()
        k3_rk4 = [p.grad.clone() for p in self.model.parameters()]
        
        # Reset and move to k4
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k3_rk4):
            p.data -= k *12
        loss = self.compute_loss(x, y)
        loss.backward()
        k4_rk4 = [p.grad.clone() for p in self.model.parameters()]
        
        # Chebyshev steps
        # Reset for Chebyshev
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
            
        # Chebyshev nodes
        pi = torch.tensor(math.pi)
        c1 = 12 * (1 + torch.cos(3*pi/8))/2
        c2 = 12 * (1 + torch.cos(pi/8))/2
        c3 = 12 * (1 - torch.cos(pi/8))/2
        
        # k2 Cheb
        for p, k in zip(self.model.parameters(), k1):
            p.data -= k * c1
        loss = self.compute_loss(x, y)
        loss.backward()
        k2_cheb = [p.grad.clone() for p in self.model.parameters()]
        
        # k3 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k2_cheb):
            p.data -= k * c2
        loss = self.compute_loss(x, y)
        loss.backward()
        k3_cheb = [p.grad.clone() for p in self.model.parameters()]
        
        # k4 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k3_cheb):
            p.data -= k * c3
        loss = self.compute_loss(x, y)
        loss.backward()
        k4_cheb = [p.grad.clone() for p in self.model.parameters()]

        # Combine RK4 result
        rk4_update = []
        for k1_p, k2_p, k3_p, k4_p in zip(k1, k2_rk4, k3_rk4, k4_rk4):
            update = (k1_p + 2*k2_p + 2*k3_p + k4_p)/6
            rk4_update.append(update)
            
        # Combine Cheb result
        sqrt_5 = torch.sqrt(torch.tensor(5.0))
        w1 = w2 = (18 + 5*sqrt_5)/72
        w3 = w4 = (18 - 5*sqrt_5)/72
        cheb_update = []
        for k1_p, k2_p, k3_p, k4_p in zip(k1, k2_cheb, k3_cheb, k4_cheb):
            update = (w1*k1_p + w2*k2_p + w3*k3_p + w4*k4_p)/6
            cheb_update.append(update)

        # Average the two methods and apply final clipping
        for i, (p, rk4_u, cheb_u) in enumerate(zip(self.model.parameters(), rk4_update, cheb_update)):
            update = (rk4_u + cheb_u)/2
            p.data -= update * 12
            p.grad.zero_()
            
        final_loss = self.compute_loss(x, y)
        return final_loss.item()


from torch.optim.optimizer import Optimizer
import torch
import numpy as np

class RCO_Batching(Optimizer):
    """
    Runge-Kutta-Chebyshev Optimizer (RCO) with batching support and scheduler compatibility.
    Uses fixed numerical coefficients and combined RK4-Chebyshev method.
    """
    def __init__(self, params, loss_function=None, max_batch_size=None):
        defaults = dict()
        super(RCO_Batching, self).__init__(params, defaults)
        
        self.max_batch_size = max_batch_size
        self.loss_function = loss_function or torch.nn.CrossEntropyLoss()
        
        # Initialize step counter for scheduler compatibility
        self.state['step'] = 0
    
    def _split_batch(self, x, y):
        """Split large batches into smaller ones if needed"""
        if self.max_batch_size is None or x.size(0) <= self.max_batch_size:
            return [(x, y)]
            
        num_splits = (x.size(0) + self.max_batch_size - 1) // self.max_batch_size
        x_splits = torch.split(x, self.max_batch_size)
        y_splits = torch.split(y, self.max_batch_size)
        return list(zip(x_splits, y_splits))

    def step(self, closure=None):
        """
        Performs a single optimization step using combined RK4-Chebyshev method.
        
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Store original params
                orig_param = p.data.clone()
                
                # Initial k1
                k1 = p.grad.clone()
                
                # RK4 steps
                # k2 position
                p.data.add_(k1, alpha=-6)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k2_rk4 = p.grad.clone()
                p.data.copy_(orig_param)
                
                # k3 position
                p.data.add_(k2_rk4, alpha=-4)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k3_rk4 = p.grad.clone()
                p.data.copy_(orig_param)
                
                # k4 position
                p.data.add_(k3_rk4, alpha=-12)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k4_rk4 = p.grad.clone()
                p.data.copy_(orig_param)
                
                # Chebyshev steps
                # Chebyshev nodes
                pi = torch.tensor(math.pi)
                c1 = 12 * (1 + torch.cos(3*pi/8))/2
                c2 = 12 * (1 + torch.cos(pi/8))/2
                c3 = 12 * (1 - torch.cos(pi/8))/2
                
                # k2 Cheb
                p.data.add_(k1, alpha=-c1)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k2_cheb = p.grad.clone()
                p.data.copy_(orig_param)
                
                # k3 Cheb
                p.data.add_(k2_cheb, alpha=-c2)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k3_cheb = p.grad.clone()
                p.data.copy_(orig_param)
                
                # k4 Cheb
                p.data.add_(k3_cheb, alpha=-c3)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k4_cheb = p.grad.clone()
                
                # Combine RK4 result
                rk4_update = (k1 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4)/6
                
                # Combine Cheb result
                sqrt_5 = torch.sqrt(torch.tensor(5.0))
                w1 = w2 = (18 + 5*sqrt_5)/72
                w3 = w4 = (18 - 5*sqrt_5)/72
                cheb_update = (w1*k1 + w2*k2_cheb + w3*k3_cheb + w4*k4_cheb)/6
                
                # Average the updates and apply
                update = (rk4_update + cheb_update)/2
                p.data = orig_param - update * 12
                p.grad.zero_()

        return loss


