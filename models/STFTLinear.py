import torch
import torch.nn as nn
from torch.functional import F
from utils.time_freq import (
    time_to_timefreq, timefreq_to_time, zero_pad_high_freq, zero_pad_low_freq
)
from typing import Tuple
from einops import rearrange


# TODO - enable no LF-HF separation

class Model(nn.Module):
    """
    Time Series Forecasting on STFT time-frequency domain. \n
    Applies a linear model for each frequency channel (bin) of STFT
    to predict the future, and restores the prediction
    to the time domain using inverse STFT.

    Notes
    -----
    Same linear model shared for each frequency bin
    to reduce computational cost.
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
            - `independent_freqs` (bool): Whether to use
              separate linear model for each frequency component.
            
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.nfft = configs.nfft
        self.n_freqs = self.nfft // 2 + 1
        self.hop_length = configs.stft_hop_length
        if self.hop_length >= self.nfft:
            raise AssertionError(
                'stft_hop_length must be shorter than nfft'
            )
        self.individual = configs.individual
        self.independent_freqs = configs.independent_freqs

        self.latent_input_len, self.latent_pred_len = self._get_latent_length()

        # Linear layers
        if self.individual and self.independent_freqs:  
            # One model per channel per frequency
            self.Linear_Freq_Latent = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.latent_input_len, self.latent_pred_len)
                    for _ in range(2 * self.in_channels)
                ]) for _ in range(self.n_freqs)
            ])
        elif self.individual: # One model per channel
            self.Linear_Freq_Latent = nn.ModuleList([
                nn.Linear(self.latent_input_len, self.latent_pred_len)
                for _ in range(2 * self.in_channels)
            ])
        elif self.independent_freqs: 
            # One model per frequency, shared among all channels
            self.Linear_Freq_Latent = nn.ModuleList([
                nn.Linear(self.latent_input_len, self.latent_pred_len)
                for _ in range(self.n_freqs)
            ])
        else:  # One shared model for all channels
            self.Linear_Freq_Latent = nn.Linear(
                self.latent_input_len, self.latent_pred_len)

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
        C = x.shape[1]
        xf = time_to_timefreq(
            x, self.nfft, C,
            stft_kwargs={'hop_length': self.hop_length}) # [b, c, f, t]

        u_l = zero_pad_high_freq(xf) # [b, c, f, t]
        u_h = zero_pad_low_freq(xf) # [b, c, f, t]
        
        ### Latent Prediction ###
        if self.independent_freqs:
            u_l_pred = []
            u_h_pred = []
            
            # Loop through each frequency component
            for f in range(self.n_freqs):
                if self.individual:
                    # Use separate model for each channel at this frequency
                    u_l_pred.append([
                        self.Linear_Freq_Latent[f][i](u_l[:, i, f, :])
                        for i in range(2 * self.in_channels)
                    ])
                    u_h_pred.append([
                        self.Linear_Freq_Latent[f][i](u_h[:, i, f, :])
                        for i in range(2 * self.in_channels)
                    ])
                else:
                    # Use shared model for this frequency component
                    u_l_pred.append(self.Linear_Freq_Latent[f](u_l[:, :, f, :]))
                    u_h_pred.append(self.Linear_Freq_Latent[f](u_h[:, :, f, :]))
            
            # Stack the predictions along the channels
            if self.individual:
                u_l_pred = [torch.stack(preds, dim=1) for preds in u_l_pred]
                u_h_pred = [torch.stack(preds, dim=1) for preds in u_h_pred]
            
            # stack along frequency dimension
            u_l_pred = torch.stack(u_l_pred, dim=2)
            u_h_pred = torch.stack(u_h_pred, dim=2)
        
        else:
            # Use a single model for all frequencies
            if self.individual:
                u_l_pred = [
                    self.Linear_Freq_Latent[i](u_l[:, i, :, :])
                    for i in range(2 * self.in_channels)]
                u_h_pred = [
                    self.Linear_Freq_Latent[i](u_h[:, i, :, :])
                    for i in range(2 * self.in_channels)]
                u_l_pred = torch.stack(u_l_pred, dim=1)
                u_h_pred = torch.stack(u_h_pred, dim=1)
            else:
                # Use one shared model for all frequencies and channels
                u_l_pred = self.Linear_Freq_Latent(u_l.flatten(0, 1))
                u_h_pred = self.Linear_Freq_Latent(u_h.flatten(0, 1))

                # Reshape back to [batch_size, channels, frequency_bins, latent_pred_len]
                u_l_pred = u_l_pred.view(
                    u_l.shape[0], 2 * self.in_channels, u_l.shape[2], self.latent_pred_len)
                u_h_pred = u_h_pred.view(
                    u_h.shape[0], 2 * self.in_channels, u_h.shape[2], self.latent_pred_len)

        # Restore predictions to the time domain using inverse STFT
        x_l = timefreq_to_time(u_l_pred, self.nfft, C,
                               stft_kwargs={'hop_length': self.hop_length})
        x_h = timefreq_to_time(u_h_pred, self.nfft, C,
                               stft_kwargs={'hop_length': self.hop_length})
        # compensate for reduction in length
        x_l = F.interpolate(x_l, self.pred_len, mode='linear')
        x_h = F.interpolate(x_h, self.pred_len, mode='linear')

        return rearrange(x_l + x_h, 'b c l -> b l c')

    def _get_latent_length(self) -> Tuple[int, int]:
        """
        Compute the latent length for both the input sequence
        and the prediction horizon.
        
        Returns:
            tuple: (latent_length_input, latent_length_pred)
        """
        batch_size = 1
        fake_data = torch.randn(batch_size, self.in_channels, self.seq_len)
        
        # Compute the time-frequency representation using time_to_timefreq
        time_freq_data = time_to_timefreq(
            fake_data, self.nfft, self.in_channels,
            stft_kwargs={'hop_length': self.hop_length})
        
        fake_data = torch.randn(batch_size, self.in_channels, self.pred_len)

        latent_length_input = time_freq_data.size(3)

        time_freq_data = time_to_timefreq(
            fake_data, self.nfft, self.in_channels,
            stft_kwargs={'hop_length': self.hop_length})
        
        latent_length_pred = time_freq_data.size(3)
        # print(f'latent shape of prediction horizon {time_freq_data.shape}')
        
        return latent_length_input, latent_length_pred
