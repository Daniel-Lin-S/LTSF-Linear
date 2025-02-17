from torch import nn
import torch
from typing import Tuple, List

    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size: int, stride: int):
        """
        Parameters
        ----------
        kernel_size: int
            the size of the averaging window
        stride: int
            distance between starting point
            of adjacent windows. 
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        """
        Compute a smooth version of x

        Parameter
        ---------
        x : torch.Tensor
            of shape (batch_size, length, channels)

        Return
        ------
        torch.Tensor
            of the same shape as input
        """
        # padding on the both ends of time series to preserve length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # take the average on temporal axis
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeasonTrendDecomp(nn.Module):
    """
    Season-trend decomposition.
    """
    def __init__(self, kernel_size: int):
        """
        Parameters
        ----------
        kernel_size: int
            the size of the averaging window.
        """
        super(SeasonTrendDecomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            of shape (batch_size, length, channels)
        
        Return
        ------
        Tuple[torch.Tensor, torch.Tensor]
            the season and trend components of the input
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiSeasonTrendDecomp(nn.Module):
    """
    Season-trend decomposition with multiple kernels.
    The moving averaged series with different kernel sizes
    are combined with a learnable linear layer to get the final trend series.
    """
    def __init__(self, kernel_size: List[int]):
        """
        Parameters
        ----------
        kernel_size: List[int]
            the sizes of the averaging windows.
        """
        super(MultiSeasonTrendDecomp, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        # learnable weights for the moving average series
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            of shape (batch_size, length, channels)
        
        Return
        ------
        Tuple[torch.Tensor, torch.Tensor]
            the season and trend components of the input
        """
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(
            moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 
