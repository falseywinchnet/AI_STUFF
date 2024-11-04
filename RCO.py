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



from torch.optim.optimizer import Optimizer
import torch
import math

class RCO_Batching(Optimizer):
    """
    Runge-Kutta-Chebyshev Optimizer (RCO) with batching support and scheduler compatibility
    """
    def __init__(self, params, loss_function=None, lr=1e-3, eps=1e-8, 
                 accumulation_steps=1, max_batch_size=None):
        defaults = dict(lr=lr, eps=eps)
        super(RCO, self).__init__(params, defaults)
        
        self.accumulation_steps = accumulation_steps
        self.max_batch_size = max_batch_size
        self.loss_function = loss_function or torch.nn.CrossEntropyLoss()
        
        # Initialize step counter for scheduler compatibility
        self.state['step'] = 0
        
        # Pre-compute tensor constants on cuda if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pi_tensor = torch.tensor(math.pi, device=device)
        
        # Chebyshev weights (convert to tensors)
        sqrt_5 = torch.sqrt(torch.tensor(5.0, device=device))
        self.w1 = self.w2 = (18 + 5*sqrt_5)/72
        self.w3 = self.w4 = (18 - 5*sqrt_5)/72

    def compute_loss(self, x, y):
        """Compute loss for a batch of data"""
        outputs = self.model(x)
        return self.loss_function(outputs, y)
    
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
        Performs a single optimization step.
        
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
                
                effective_lr = group['lr'] / self.accumulation_steps
                
                # Store original params
                orig_param = p.data.clone()
                
                # Initial k1
                k1 = p.grad.clone()
                
                # RK4 steps
                # k2 position
                p.data.add_(k1, alpha=-effective_lr/2)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k2_rk4 = p.grad.clone()
                p.data.copy_(orig_param)
                
                # k3 position
                p.data.add_(k2_rk4, alpha=-effective_lr/3)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k3_rk4 = p.grad.clone()
                p.data.copy_(orig_param)
                
                # k4 position
                p.data.add_(k3_rk4, alpha=-effective_lr)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k4_rk4 = p.grad.clone()
                
                # Chebyshev steps
                # Compute nodes
                c1 = effective_lr * (1 + torch.cos(3*self.pi_tensor/8))/2
                c2 = effective_lr * (1 + torch.cos(self.pi_tensor/8))/2
                c3 = effective_lr * (1 - torch.cos(self.pi_tensor/8))/2
                
                # k2 Cheb
                p.data.copy_(orig_param)
                p.data.add_(k1, alpha=-c1)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k2_cheb = p.grad.clone()
                
                # k3 Cheb
                p.data.copy_(orig_param)
                p.data.add_(k2_cheb, alpha=-c2)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k3_cheb = p.grad.clone()
                
                # k4 Cheb
                p.data.copy_(orig_param)
                p.data.add_(k3_cheb, alpha=-c3)
                if closure is not None:
                    with torch.enable_grad():
                        closure()
                k4_cheb = p.grad.clone()
                
                # Combine RK4 result
                rk4_update = (k1 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4)/6
                
                # Combine Cheb result
                cheb_update = (self.w1*k1 + self.w2*k2_cheb + self.w3*k3_cheb + self.w4*k4_cheb)
                
                # Average the updates and apply
                update = (rk4_update + cheb_update)/2
                p.data = orig_param - effective_lr * update

        return loss

    def get_lr(self):
        """Returns current learning rate for scheduler compatibility"""
        return [group['lr'] for group in self.param_groups]


for minst:

loss_function = nn.CrossEntropyLoss()
optimizer = RCO(
    params=net.parameters(),
    loss_function=loss_function,  # Your cross entropy loss
    lr=0.5,
    accumulation_steps=1,  # Adjust based on your memory constraints
    max_batch_size=batch_size  # Use your existing batch size
)

# Training loop
losses = []
for i, (images, labels) in enumerate(train_gen):
        # Move data to GPU
        images = images.view(-1, 28*28).cuda()
        labels = labels.cuda()
        
        def closure():
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            return loss
        
        # Compute loss and update in one step
        loss = optimizer.step(closure)
        losses.append(loss.item())   
    
plt.plot(losses)

ie everything happens inside the model

\


class MomentumPIDScheduler(_LRScheduler):
    """
    PID Learning Rate Scheduler with Loss Momentum and Nesterov Look-ahead
    """
    def __init__(self, optimizer, 
                 kp=0.1, ki=0.01, kd=0.001,
                 momentum=0.9, nesterov=True,
                 min_lr=1e-6, max_lr=1.0,
                 window_size=5, verbose=False):
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.momentum = momentum
        self.nesterov = nesterov
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # State variables
        self.loss_history = []
        self.window_size = window_size
        self.integral_error = 0
        self.last_error = None
        self.velocity = 0  # momentum velocity
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch=-1, verbose=verbose)

    def compute_momentum_pid_adjustment(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        if len(self.loss_history) < 2:
            return 1.0
            
        # Compute loss change with momentum
        current_loss = self.loss_history[-1]
        prev_loss = self.loss_history[-2]
        raw_delta = (current_loss - prev_loss) / prev_loss
        
        # Apply momentum to loss delta
        self.velocity = (self.momentum * self.velocity + 
                        (1 - self.momentum) * raw_delta)
        
        # Nesterov look-ahead if enabled
        error = self.velocity
        if self.nesterov:
            error = self.velocity + self.momentum * (self.velocity - self.last_error) if self.last_error is not None else self.velocity
            
        # P term
        p_term = self.kp * error
        
        # I term with momentum decay
        self.integral_error = (self.momentum * self.integral_error + 
                             (1 - self.momentum) * error)
        i_term = self.ki * self.integral_error
        
        # D term on momentum-smoothed signal
        if self.last_error is not None:
            d_term = self.kd * (error - self.last_error)
        else:
            d_term = 0
        self.last_error = error
        
        # Combine terms and convert to multiplicative factor
        pid_output = -(p_term + i_term + d_term)
        adjustment = math.exp(pid_output)
        
        # Clip the adjustment
        return max(self.min_lr, min(self.max_lr, adjustment))

    def step(self, loss=None):
        if loss is None:
            raise ValueError("Loss value required for momentum PID scheduling")
            
        adjustment = self.compute_momentum_pid_adjustment(loss)
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.initial_lrs[i] * adjustment
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'velocity': self.velocity,
            'integral_error': self.integral_error,
            'last_error': self.last_error,
            'loss_history': self.loss_history,
            'initial_lrs': self.initial_lrs
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = MomentumPIDScheduler(
    optimizer,
    kp=0.1,
    ki=0.01,
    kd=0.001,
    momentum=0.9,
    nesterov=True
)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

note: this scheduler is a last minute conceptual idea and has not been tested. 
i simply wanted to try to see if PID theory could work instead of adam.
