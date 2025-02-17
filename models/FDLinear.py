import torch
import torch.nn as nn
from utils.time_freq import stft_lfhf
from einops import rearrange


class Model(nn.Module):
    """
    Replace the season-trend decomposition in DLinear
    with a similar decomposition using STFT. \n
    Trend component: frequency component 0 is treated as trend,
    and the other components are replaced with zeroes,
    then iSTFT is applied to obtain the trend component. \n
    Season component: The remaining components are treated as seasonality,
    and similar operations are applied to obtain the seasonality component. \n
    """

    def __init__(self, configs):
        """
        Parameters
        ----------
        configs : argparse.Namespace
            Configuration object that contains the following attributes:
            
            - `seq_len` (int): Length of the input sequence.
            - `pred_len` (int): Length of the prediction horizon.
            - `enc_in` (int): Number of input channels.
            - `stft_hop_length` (int): Hop length for the STFT transformation.
            - `individual` (bool): Whether to use separate
            linear models for each input channel.
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.nfft = configs.nfft
        self.hop_length = configs.stft_hop_length
        if self.hop_length >= self.nfft:
            raise AssertionError(
                'stft_hop_length must be shorter than nfft'
            )
        self.individual = configs.individual

        # Linear layers for frequency components
        if self.individual:
            # Shared linear model for all channels
            self.Linear_Freq = nn.Linear(self.seq_len, self.pred_len)
        else:
            # Separate linear models for each channel
            self.Linear_Freq = nn.ModuleList()
            for _ in range(self.in_channels):
                self.Linear_Freq.append(nn.Linear(self.seq_len, self.pred_len))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, seq_len, channels)

        Returns
        -------
        torch.Tensor
            The predicted values with shape (batch_size, pred_len, channels).
        """
        x = rearrange(x, 'b l c -> b c l')
        
        # Decompose into time-frequency domain using STFT
        x_l, x_h = stft_lfhf(x, self.nfft,
                             stft_kwargs={'hop_length': self.hop_length})
        
        # Prediction for each frequency component
        if self.individual:
            # Apply the same linear model to all channels
            x_l = self.Linear_Freq(x_l)
            x_h = self.Linear_Freq(x_h)
        else:
            for i in range(self.in_channels):
                x_l[:, i, :] = self.Linear_Freq[i](x_l[:, i, :])
                x_h[:, i, :] = self.Linear_Freq[i](x_h[:, i, :])

        # Combine low and high frequency components
        x = x_l + x_h
        return rearrange(x, 'b c l -> b l c')
