from torch import nn
import torch
import torch.nn.functional as F
from typing import Tuple, List

from utils.time_freq import (
    time_to_timefreq, timefreq_to_time,
    zero_pad_high_freq, zero_pad_low_freq
)

    
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
    def __init__(self, kernel_size: int) -> None:
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
            the season and trend components of the input.
            Both of the same shape as input.
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


class FreqDecomp(nn.Module):
    """
    Split the input into high and low frequency components.
    """
    def __init__(
            self, nfft: int, hop_length: int,
            n_low_freqs: int = 1,
            split_trend: bool = False
            ):
        """
        Parameters
        ----------
        nfft: int
            the size of the FFT window.
        hop_length: int
            the size of the hop length.
        n_low_freqs: int, optional
            The number of low frequency components to keep.
            Default is 1.
        split_trend: bool, optional
            If true, only perform frequency decomposition
            on the season component.
        """
        super(FreqDecomp, self).__init__()
        if hop_length >= nfft:
            raise AssertionError(
                'hop_length must be shorter than nfft')
        
        if n_low_freqs >= nfft // 2 + 1:
            raise AssertionError(
                'n_low_freqs must be shorter than nfft // 2 '
                f'(total number of frequency components) {nfft // 2 + 1}')
        elif n_low_freqs < 1:
            raise AssertionError(
                'n_low_freqs must be at least 1')

        self.nfft = nfft
        self.hop_length = hop_length
        self.n_low_freqs = n_low_freqs

        if split_trend:
            self.st_decomp = SeasonTrendDecomp(kernel_size=25)
        else:
            self.st_decomp = None
    
    def forward(self,
            x: torch.Tensor
        ) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            of shape (batch_size, length, channels)
        
        Return
        ------
        Tuple[torch.Tensor]
            If split_trend is False,
            return the low and high frequency components.
            Otherwise, return the low frequency, high frequency,
            and the trend component.
        """
        in_channels = x.shape[2]
        input_length = x.shape[1]

        if self.st_decomp:
            x_season, x_trend = self.st_decomp(x)
        else:
            x_season = x  # (b l c)

        x_season = x_season.permute(0, 2, 1)  # (b c l)
        xf = time_to_timefreq(x_season, self.nfft, in_channels)  # (b 2c h w)

        u_l = zero_pad_high_freq(xf, self.n_low_freqs)  # (b 2c h w)
        x_l = F.interpolate(
            timefreq_to_time(u_l, self.nfft, in_channels),
            input_length, mode='linear')  # (b c l)
        u_h = zero_pad_low_freq(xf, self.n_low_freqs)  # (b 2c h w)
        x_h = F.interpolate(
            timefreq_to_time(u_h, self.nfft, in_channels),
            input_length, mode='linear')  # (b c l)
        
        x_l = x_l.permute(0, 2, 1)  # (b l c)
        x_h = x_h.permute(0, 2, 1)  # (b l c)

        if self.st_decomp:
            return x_l, x_h, x_trend
        else:
            return x_l, x_h
