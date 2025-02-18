import torch.jit as jit
import torch.nn as nn
import torch
import numpy as np


class SnakeActivation(jit.ScriptModule):
    """
    Snake activation: a non-linear activation
    that adds sine-squared to the input. \n
    The activation function is defined as:
    
        f(x) = x + (1 / a) * sin(a * x)^2

    This implementation allows multiple values of `a`
    for different channels. (num_features)

    Attributes
    ----------
    a : torch.Tensor
        The learnable or fixed parameter controlling the
        frequency and amplitude of the sinusoidal component. \n
        Its shape depends on the input dimension (`dim`).
    """
    def __init__(self, num_features:int, dim:int, a_base=0.2, learnable=True, a_max=0.5):
        """
        Parameters
        ----------
        num_features : int
            The number of features (channels) in the input data.
        dim : int
            The dimensionality of the input data. \n
            Must be 1 (e.g., time series) or 2 (e.g., images).
        a_base : float, optional
            The fixed value for the parameter `a` when `learnable=False`
            or as the lower bound for random initialization when
            `learnable=True`. \n
            Default is 0.2.
        learnable : bool, optional
            Whether the parameter `a` should be learnable. \n
            Default is True.
        a_max : float, optional
            The upper bound for random initialization of `a`
            when `learnable=True`. \n
            Default is 0.5.
        """
        super().__init__()
        assert dim in [1, 2], '`dim` supports 1D and 2D inputs.'

        # prepare the parameter a
        if learnable:
            if dim == 1:  # input has shape (b num_features l); like time series
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1))  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2: # input has shape (b num_features h w); like image
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1, 1))  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else: # fixed value
            self.register_buffer('a', torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            must be of shape (batch_size, num_features, length)
            or (batch_size, num_features, h, w)
            where h w are spatial dimensions
            when input is an image.
        """
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2