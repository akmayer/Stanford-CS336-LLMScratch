import torch
import einx
from collections.abc import Callable, Iterable
from typing import Optional
import math

def cross_entropy(inputs, targets):
    """
        Given a tensor of inputs and targets, compute the average cross-entropy
        loss across examples.
    
        Args:
            inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
            targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.
    
        Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    inputs = einx.rearrange("... v -> (...) v", inputs)
    targets = einx.rearrange("... -> (...)", targets)
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    trueLogits = einx.get_at("b [v], (b [idx]) -> b 1", inputs, targets)
    eLogits = torch.exp(inputs)
    logSumExp = torch.log(eLogits.sum(axis=-1, keepdim=True))
    return torch.mean(- (trueLogits - logSumExp))


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), weight_decay = 0.01, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, 
                    "beta_1" : betas[0], 
                    "beta_2" : betas[1],
                   "weight_decay" : weight_decay,
                   "epsilon" : eps}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1)
                m = state.get("firstMoment", torch.zeros_like(p.data)) # Get iteration number from the state, or initial value.
                v = state.get("secondMoment", torch.zeros_like(p.data)) # Get iteration number from the state, or initial value.

                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad ** 2
                adjustedLR = lr * ((1 - beta_2 ** t) ** 0.5) / (1 - beta_1 ** t)
                p.data -= adjustedLR * m / (v ** 0.5 + epsilon)
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['firstMoment'] = m
                state['secondMoment'] = v
                
        return loss

def lrScheduler(t, alphaMax, alphaMin, Tw, Tc):
    if t < Tw:
        return t / Tw * alphaMax
    elif t <= Tc:
        cosPortion = math.cos((t - Tw) / (Tc - Tw) * math.pi)
        return alphaMin + 0.5 * (1 + cosPortion) * (alphaMax - alphaMin)
    else:
        return alphaMin