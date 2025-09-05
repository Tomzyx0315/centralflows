from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .utils import flatten_pytree


Array = Any


class UpdateRule:
    """Abstract class for an update rule that defines an optimization algorithm.
    
    Specifically, this is for optimization algorithms that perform preconditioned
    gradient descent:
           w_{t+1} = w_t - P^{-1}_t ∇L(w_t)
    where the preconditioner P_t depends on some evolving state.
    
    This formulation encompasses gradient descent with a fixed learning rate (a trivial special case 
    with P_t = 1/η I), gradient descent with a learning rate schedule, Scalar RMSProp, and RMSProp.
    
    The functional design here is inspired by Jax's Optax library.
    """
        
    def initialize_state(self, w: torch.Tensor) -> Array:
        """Initialize the state.
        
        Args:
          w: the initial weights
          
        Returns:
          Array: the state, as a flat vector (see subclasses for examples)
        """
        raise NotImplementedError()

    def P(self, flat_state: Array) -> Preconditioner:
        """Return the current preconditioner.
        
        Args:
          flat_state (Array): the current state, as a flat vector
        
        Returns:
          Preconditioner: the current preconditioner
        """
        raise NotImplementedError()

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """Update the state (discrete time).
        
        Args:
          flat_state (Array): the current state, as a flat vector
          gradient (Array): the current gradient
        
        Returns:
          Array: the next state
        """
        raise NotImplementedError()

    def dstate_dt(self, flat_state: Array, gradient: Array):
        """Update the state (continuous time).
        
        Args:
          flat_state (Array): the current state, as a flat vector
          gradient (Array): the current gradient
        
        Returns:
          Array: the time derivative of the state
        """
        return self.update_state(flat_state, gradient) - flat_state

    def summarize_state(self, flate_state: Array) -> Dict[str, Any]:
        """Summarize the state.
        
        Args:
          flat_state (Array): the current state, as a flat vector
          
        Returns:
          Dict[str, Any]: a dictionary with a summary of the current state.
        """
        raise NotImplementedError()
    
    def update(self, w: Array, flat_state: Array, gradient: Array) -> Tuple[Array, Array]:
        """Update both the weights and optimizer state."""
        flat_state = self.update_state(flat_state, gradient)
        w = w - self.P(flat_state).pow(-1)(gradient)
        return w, flat_state


@dataclass # we make it a dataclass so that it can be instantiated from the command line
class GradientDescent(UpdateRule):
    """Gradient descent with a fixed or scheduled learning rate: 
             w_{t+1} = w_t - η(t) ∇L(w_t)
    where η(t) is the learning rate at step t.
    
    This is a special case of `UpdateRule` with P_t set to P_t = 1/η(t) I.
    
    The state consists of the current step counter t (to support learning rate schedules).
    """
    lr: float  # the learning rate

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array) -> Array:
        state = {"t": torch.tensor(0.0, dtype=w.dtype, device=w.device)}
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return DiagonalPreconditioner(1 / self.lr_fn(state["t"]))

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        state = {"t": state["t"] + 1.0}
        return flatten_pytree(state)[0]

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return {
            "t": state["t"],
            "lr": self.lr_fn(state["t"]),
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        """Given the top eigenvalues of the effective Hessian, return the top 
        eigenvalues of the 'raw' Hessian.
        
        This function is used by the 'raw' Hessian eigenvalue logger.
        
        Args:
          flate_state (Array): the current state, as a flattened vector
          eigs (Array): the top eigenvalues of the effective Hessian
          
        Returns:
          (Array): the top eigenvalues of the 'raw' Hessian
        """
        if eigs is None:
            return None
        lr = self.P(flat_state).pow(-1)(1.0)
        return eigs / lr


@dataclass # we make it a dataclass so that it can be instantiated from the command line
class ScalarRMSProp(UpdateRule):
    """The Scalar RMSProp optimizer.
    
    This optimizer maintains an EMA ν of the squared gradient norm,
    and takes gradient steps using the effective step size η/sqrt(ν).
    Our implementation supports optional learning rate scheduling,
    bias correction, and ε:
    
           ν_{t} = (1 - β_2) ν_{t-1} + β_2 ||∇L(w_t)||^2
           ν̂_{t} = ν_t / (1 - β_2 ^ t)
           w_{t+1} = w_t - η(t) / sqrt (ν̂_t + ε) * ∇L(w_t)
           
    The optimizer's state consists of the tuple (t, ν).
    """
    
    lr: float
    beta2: float
    eps: float = 0.
    bias_correction: bool = False

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "nu": torch.tensor(0.0, dtype=w.dtype, device=w.device),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        if self.bias_correction:
            nu_hat =  (nu / (1 - self.beta2**(t)))
        else:
            nu_hat = nu
        lrs = self.lr_fn(t) / (torch.sqrt(nu_hat) + self.eps)
        return DiagonalPreconditioner(1 / lrs)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        nu = nu + (1 - self.beta2) * (gradient.square().sum() - nu)
        state = {"t": t + 1.0, "nu": nu}
        return flatten_pytree(state)[0]

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        update = self.update_state(flat_state, gradient) - flat_state
        return update / self.beta2  # see footnote TODO in paper for explanation of this

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        ess = self.P(flat_state).pow(-1)(1.0)
        return {
            "t": state["t"],
            "nu": state["nu"],
            "lr": self.lr_fn(state["t"]),
            "effective_step_size": ess,
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        """Given the top eigenvalues of the effective Hessian, return the top 
        eigenvalues of the 'raw' Hessian.
        
        This function is used by the 'raw' Hessian eigenvalue logger.
        
        Args:
          flate_state (Array): the current state, as a flattened vector
          eigs (Array): the top eigenvalues of the effective Hessian
          
        Returns:
          (Array): the top eigenvalues of the Hessian
        """
        if eigs is None:
            return None
        ess = self.P(flat_state).pow(-1)(1.0) # effective step size
        return eigs / ess


@dataclass # we make it a dataclass so that it can be instantiated from the command line
class RMSProp(UpdateRule):
    """The RMSProp optimizer.
    
    This optimizer maintains an EMA ν of the elementwise squared gradient, 
    and takes gradient steps using the effective step sizes η/sqrt(ν).
    Our implementation supports optional learning rate scheduling,
    bias correction, and ε:
    
           ν_{t} = (1 - β_2) ν_{t-1} + β_2 ∇L(w_t)^2
           ν̂_{t} = ν_t / (1 - β_2 ^ t)
           w_{t+1} = w_t - η(t) / sqrt (ν̂_t + ε) * ∇L(w_t)
           
    The optimizer's state consists of the tuple (t, ν).
    """
    
    lr: float
    beta2: float
    eps: float = 0
    bias_correction: bool = False

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "nu": torch.zeros_like(w),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        if self.bias_correction:
            nu_hat =  (nu / (1 - self.beta2**(t)))
        else:
            nu_hat = nu
        lrs = self.lr_fn(t) / (torch.sqrt(nu_hat) + self.eps)
        return DiagonalPreconditioner(1 / lrs)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        nu = nu + (1 - self.beta2) * (gradient.square() - nu)
        state = {"t": t + 1.0, "nu": nu}
        return flatten_pytree(state)[0]

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        update = self.update_state(flat_state, gradient) - flat_state
        return update / self.beta2 # see footnote TODO in paper for explanation of this

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        nu = state["nu"]
        ess = self.P(flat_state).pow(-1)(torch.ones_like(nu))
        selected_idx = np.linspace(0, len(nu) - 1, 25, dtype=int)
        return {
            "t": state["t"],
            "nu_l1": nu.sum(),                   # L1 norm of nu
            "nu_selected_idx": nu[selected_idx], # selected indices of nu
            "ess_mean": ess.mean(),              # mean of effective step sizes
                                                 # harmonic mean of effective step sizes
            "ess_harmonic_mean": ess.reciprocal().mean().reciprocal(),
            "lr": self.lr_fn(state["t"]), # current learning rate
        }


class Preconditioner:
    """Abstract class for a preconditioner."""
    
    def __call__(self, v: Array) -> Array:
        """Precondition a vector.
        
        Args:
          v: the vector to precondition
          
        Returns:
          the preconditioned vector Pv
        """
        raise NotImplementedError()
    
    def pow(self, p: float) -> Preconditioner:
        """Return a new preconditioner which is this preconditioner raised to a power.
        
        Args:
          p: the power
        
        Returns:
          (Preconditioner): a new preconditioner
        """
        raise NotImplementedError()


class DiagonalPreconditioner(Preconditioner):
    """A diagonal (i.e. elementwise) preconditioner."""
    
    def __init__(self, P):
        """Constructor for the diagonal preconditioner.
        
        Args:
          P (Array): the diagonal preconditioner, as a vector
        """
        self.P = P

    def __call__(self, v: Array) -> Array:
        return v * self.P

    def pow(self, power: float) -> DiagonalPreconditioner:
        return DiagonalPreconditioner(self.P**power)


def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          # if it's a schedule ...
        return schedule_or_constant             #  ... do nothing.
    else:                                       # but if it's a constant...
        return lambda t: schedule_or_constant   # ... turn it into a schedule. 
