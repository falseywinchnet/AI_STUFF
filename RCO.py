"""
      Runge-Kutta-Chebyshev Optimizer (RCO) 2024 copyright joshuah rainstar joshuah.rainstar@gmail.com
      version 1.0 (11/4/24)
      licensed under MIT license
"""

from torch.optim import Optimizer
class RCO_new(Optimizer):
    """
      Runge-Kutta-Chebyshev Optimizer (RCO) - A neural network optimizer that combines
      4th order Runge-Kutta method and Chebyshev polynomial interpolation for simulated annealing

      This optimizer implements a novel approach to neural network optimization by:
      1. Computing gradients using both RK4 and Chebyshev methods
      2. Averaging the results of both methods
      3. Applying the combined update to the model parameters

      The optimizer uses four evaluation points for both RK4 and Chebyshev methods,
      providing potentially better convergence properties than traditional first-order
      methods. Repeated iterations on the same item converge on the global minima- the more iterations, the better.

      Parameters
      ----------
      model : torch.nn.Module
          The neural network model to optimize
      dt : float
            integration coefficient- the higher, the steeper and greater convergence
            chose the highest value that is still stable- recommend 12-18



      Methods
      -------
      compute_loss(x, y)
          Computes the MSE loss between model predictions and target values
      step(x, y)
          Performs a single optimization step using the combined RK4-Chebyshev method
          Call this method repeatedly on the same item for better training!
      backpropagate(x, y)
          Computes gradients for the loss function

      Example
      -------
      >>> model = torch.nn.Linear(10, 1)
      >>> optimizer = RCO(model, lr=1e-3)
      >>> for epoch in range(num_epochs):
      ...     loss = optimizer.step(x_batch, y_batch)
     """
    def __init__(self, model, dt=18):
            self.model = model
            self.dt = dt

    def compute_loss(self, x, y):
        y_pred = self.model(x)
        return torch.mean((y_pred - y)**2)

    def backpropagate(self, x,y):
           loss = self.compute_loss(x, y)
           loss.backward()  # Compute gradients for the loss

    def step(self, x, y):
      # Initial k1
      self.backpropagate(x,y)
      k1 = [p.grad.clone() for p in self.model.parameters()]
      # Store original params
      orig_params = [p.data.clone() for p in self.model.parameters()]
      
      for p, k in zip(self.model.parameters(), k1):
          p.data -= k * ( self.dt/2)
      self.backpropagate(x,y)
      k2_rk4 = [p.grad.clone() for p in self.model.parameters()]

      # Reset and move to k3 position
      for p, orig in zip(self.model.parameters(), orig_params):
          p.data.copy_(orig)
      for p, k in zip(self.model.parameters(), k2_rk4):
          p.data -= k *  (self.dt/3)
      self.backpropagate(x,y)
      k3_rk4 = [p.grad.clone() for p in self.model.parameters()]

      # Reset and move to k4
      for p, orig in zip(self.model.parameters(), orig_params):
          p.data.copy_(orig)
      for p, k in zip(self.model.parameters(), k3_rk4):
          p.data -= k *  self.dt
      self.backpropagate(x,y)
      k4_rk4 = [p.grad.clone() for p in self.model.parameters()]

      # Chebyshev steps
      # Reset for Chebyshev
      for p, orig in zip(self.model.parameters(), orig_params):
          p.data.copy_(orig)

      # Chebyshev nodes
      c1 = (self.dt * 1/11)
      c2 =  (self.dt *  2/11)
      c3 =  (self.dt *  3/11)
      c4 =  (self.dt *   4/11)

      # k2 Cheb
      for p, k in zip(self.model.parameters(), k1):
          p.data -= k * c1
      self.backpropagate(x,y)
      k2_cheb = [p.grad.clone() for p in self.model.parameters()]

      # k3 Cheb
      for p, orig in zip(self.model.parameters(), orig_params):
          p.data.copy_(orig)
      for p, k in zip(self.model.parameters(), k2_cheb):
          p.data -= k * c2
      self.backpropagate(x,y)
      k3_cheb = [p.grad.clone() for p in self.model.parameters()]

      # k4 Cheb
      for p, orig in zip(self.model.parameters(), orig_params):
          p.data.copy_(orig)
      for p, k in zip(self.model.parameters(), k3_cheb):
          p.data -= k * c3
      self.backpropagate(x,y)
      k4_cheb = [p.grad.clone() for p in self.model.parameters()]

      # k5 Cheb 
      for p, orig in zip(self.model.parameters(), orig_params):
          p.data.copy_(orig)
      for p, k in zip(self.model.parameters(), k4_cheb):
          p.data -= k * c4
      self.backpropagate(x,y)
      k5_cheb = [p.grad.clone() for p in self.model.parameters()]

      # Combine RK4  result
      rk4_update = []
      for k1_p, k2_p, k3_p, k4_p in zip(k1, k2_rk4, k3_rk4, k4_rk4):
          update = ((k1_p +  ( self.dt/( self.dt/2))*k2_p +( self.dt/( self.dt/2))*k3_p + k4_p)/( self.dt/2))
          rk4_update.append(update)

      #print the abs of the sum of rk4_update
      # Combine Cheb result
      w1 = 0.21967452022181430116
      w2 = 0.23876183658069730087
      w3 = 0.25950763224531042672
      w4 = 0.28205601095217797125

      cb4_update = []
      for k1_p, k2_p, k3_p, k4_p, k5_p in zip(k1, k2_cheb, k3_cheb, k4_cheb, k5_cheb):
          cb4_update = (w1*k2_p + w2*k3_p + w3*k4_p + w4*k5_p)

    
      for i, (p, rk4_u, cheb_u) in enumerate(zip(self.model.parameters(), rk4_update, cb4_update)):
          update = (rk4_u  - cheb_u)/2
          p.data += update

      model.zero_grad()
      final_loss = self.compute_loss(x, y)
      return final_loss.item()




from torch.optim.optimizer import Optimizer
import torch
import math

class RCO_Batching(Optimizer):
    """
    Runge-Kutta-Chebyshev Optimizer (RCO) with batching support, featuring self-decaying learning rate
    batching necessitates lowering the learn rate- this may not beat adam
    i am still experimenting with this
    it also required clamping, depending on the problem.. YRMV
    """
    def __init__(self, model, max_batch_size=None):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size
        self.scaling_factor = 0.9

        # Pre-compute constants
        pi = torch.tensor(math.pi)


        # Trig constants
        self.w1 = 0.21967452022181430116
        self.w2 = 0.23876183658069730087
        self.w3 = 0.25950763224531042672
        self.w4 = 0.28205601095217797125

    def compute_loss(self, x, y):
        y_pred = self.model(x)
        return self.loss_fn(y_pred, y)

    def _split_batch(self, x, y):
        """Split large batches into smaller ones if needed"""
        if self.max_batch_size is None or x.size(0) <= self.max_batch_size:
            return [(x, y)]

        num_splits = (x.size(0) + self.max_batch_size - 1) // self.max_batch_size
        x_splits = torch.split(x, self.max_batch_size)
        y_splits = torch.split(y, self.max_batch_size)
        return list(zip(x_splits, y_splits))

    def step(self, x, y):
        """
        Performs a single optimization step using combined RK4-Chebyshev method.
        """
        # Initial k1
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)  # Fixed line

        k1 = [p.grad.clone() for p in self.model.parameters()]

        # Store original params
        orig_params = [p.data.clone() for p in self.model.parameters()]

        # RK4 steps
        for p, k in zip(self.model.parameters(), k1):
            p.data -= k * ((6/  x.size(0)))
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)  # Fixed line

        k2_rk4 = [p.grad.clone() for p in self.model.parameters()]

        # Reset and move to k3 position
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k2_rk4):
            p.data -= k * ((4/  x.size(0)))
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)  # Fixed line

        k3_rk4 = [p.grad.clone() for p in self.model.parameters()]

        # Reset and move to k4
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k3_rk4):
            p.data -= k * ((12/  x.size(0)))
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)  # Fixed line

        k4_rk4 = [p.grad.clone() for p in self.model.parameters()]

        # Chebyshev steps
        # Reset for Chebyshev
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)

        # Chebyshev nodes with learning rate
        c1 = ((12/  x.size(0))) *1/11
        c2 = ((12/  x.size(0))) * 2/11
        c3 = ((12/  x.size(0))) * 3/11
        c4 = ((12/  x.size(0))) * 4/11

        # k2 Cheb
        for p, k in zip(self.model.parameters(), k1):
            p.data -= k * c1
        loss = self.compute_loss(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Fixed line

        k2_cheb = [p.grad.clone() for p in self.model.parameters()]

        # k3 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k2_cheb):
            p.data -= k * c2
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Fixed line

        k3_cheb = [p.grad.clone() for p in self.model.parameters()]

        # k4 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k3_cheb):
            p.data -= k * c3
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Fixed line

        k4_cheb = [p.grad.clone() for p in self.model.parameters()]

                # k5 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k4_cheb):
            p.data -= k * c4
        loss = self.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Fixed line

        k5_cheb = [p.grad.clone() for p in self.model.parameters()]

        # Combine RK4 result
        rk4_update = []
        for k1_p, k2_p, k3_p, k4_p in zip(k1, k2_rk4, k3_rk4, k4_rk4):
            update = (k1_p + 2*k2_p + 2*k3_p + k4_p)/((6))
            rk4_update.append(update)

        # Combine Cheb result
        cheb_update = []
        for k1_p, k2_p, k3_p, k4_p in zip( k2_cheb, k3_cheb, k4_cheb,k5_cheb):
            update = (self.w1*k1_p + self.w2*k2_p + self.w3*k3_p + self.w4*k4_p)
            cheb_update.append(update)

        # Average the two methods and apply final update
        for i, (p, rk4_u, cheb_u) in enumerate(zip(self.model.parameters(), rk4_update, cheb_update)):
            update = (rk4_u + cheb_u)/2
            p.data -=( update / ((x.size(0))) )
            p.grad.zero_()

        final_loss = self.compute_loss(x, y)
        return final_loss.item()
