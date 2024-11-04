class RCO:
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
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in model.parameters()]
        self.v = [torch.zeros_like(p) for p in model.parameters()]
        self.t = 0

    def compute_loss(self, x, y):
        y_pred = self.model(x)
        return torch.mean((y_pred - y)**2)
        
    def step(self, x, y):
        self.t += 1  # Add this at the start!

        # Initial k1
        loss = self.compute_loss(x, y)
        loss.backward()
        k1 = [p.grad.clone() for p in self.model.parameters()]
        
        # Store original params
        orig_params = [p.data.clone() for p in self.model.parameters()]
        
        # RK4
        # Move to k2 position and evaluate
        for p, k in zip(self.model.parameters(), k1):
            p.data-= k /2 * self.lr
        loss = self.compute_loss(x, y)
        loss.backward()
        k2_rk4 = [p.grad.clone() for p in self.model.parameters()]
        
        # Reset and move to k3 position
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k2_rk4):
            p.data-= k /3 * self.lr

        loss = self.compute_loss(x, y)
        loss.backward()
        k3_rk4 = [p.grad.clone() for p in self.model.parameters()]
        
        # Reset and move to k4
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k3_rk4):
            p.data-= k * self.lr
        loss = self.compute_loss(x, y)
        loss.backward()
        k4_rk4 = [p.grad.clone() for p in self.model.parameters()]
        
        # Chebyshev steps
        # Reset for Chebyshev
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
            
        # Chebyshev nodes
        c1 = self.lr * (1 + np.cos(3*np.pi/8))/2
        c2 = self.lr * (1 + np.cos(np.pi/8))/2
        c3 = self.lr * (1 - np.cos(np.pi/8))/2
        
        # k2 Cheb
        for p, k in zip(self.model.parameters(), k1):
            p.data-= k * c1
        loss = self.compute_loss(x, y)
        loss.backward()
        k2_cheb = [p.grad.clone() for p in self.model.parameters()]
        
        # k3 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k2_cheb):
            p.data-= k * c2
        loss = self.compute_loss(x, y)
        loss.backward()
        k3_cheb = [p.grad.clone() for p in self.model.parameters()]
        
        # k4 Cheb
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)
        for p, k in zip(self.model.parameters(), k3_cheb):
               p.data-= k * c3
        loss = self.compute_loss(x, y)
        loss.backward()
        k4_cheb = [p.grad.clone() for p in self.model.parameters()]

        # Combine RK4 result
        rk4_update = []
        for k1_p, k2_p, k3_p, k4_p in zip(k1, k2_rk4, k3_rk4, k4_rk4):
            rk4_update.append((k1_p + 2*k2_p + 2*k3_p + k4_p)/6)  # Add normalization
            
        # Combine Cheb result
        w1 = w2 = (18 + 5*np.sqrt(5))/72
        w3 = w4 = (18 - 5*np.sqrt(5))/72
        cheb_update = []
        for k1_p, k2_p, k3_p, k4_p in zip(k1, k2_cheb, k3_cheb, k4_cheb):
            cheb_update.append((w1*k1_p + w2*k2_p + w3*k3_p + w4*k4_p)/6)
            
        # Average the two methods and update with Adam
        for i, (p, rk4_u, cheb_u) in enumerate(zip(self.model.parameters(), rk4_update, cheb_update)):
            update = (rk4_u + cheb_u)/2            
            p.data-=  update / self.lr
            p.grad.zero_()
            
        # Final loss computation for tracking
        final_loss = self.compute_loss(x, y)
        return final_loss.item()
