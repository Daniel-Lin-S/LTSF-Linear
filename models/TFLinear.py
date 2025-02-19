import torch
import torch.nn as nn
from torch.functional import F
from typing import Tuple
from einops import rearrange

from utils.time_freq import STFT
from layers.Decompositions import SeasonTrendDecomp


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
            - `nfft` (int): Number of FFT points.
            - `stft_hop_length` (int): Hop length for the STFT transformation.
            - `isolate trend` (bool): If true, isolate the trend component
            - `individual` (bool): Whether to use separate
              linear models for each input channel.
            - `independent_freqs` (bool): Whether to use
              separate linear model for each frequency component.        
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.requires_time_markers = False

        self.n_freqs = configs.nfft // 2 + 1
        self.individual = configs.individual
        self.independent_freqs = configs.independent_freqs
        self.isolate_trend = configs.isolate_trend

        self.stft = STFT(
            n_fft=configs.nfft, hop_length=configs.stft_hop_length)

        # linear models for trend component
        if self.isolate_trend:
            self.decomp = SeasonTrendDecomp(kernel_size=25)
            if self.individual:
                self.Linear_Trend = nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len)
                    for _ in range(self.in_channels)])
            else:
                self.Linear_Trend = nn.Linear(
                    self.seq_len, self.pred_len)

        self.latent_input_len, self.latent_pred_len = self._get_latent_length()

        # Linear layers for time-frequency graph
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
        
        # Decompose into time-frequency domain using STFT
        if self.isolate_trend:
            x_s, x_trend = self.decomp(x)

            if self.individual:
                pred_trend = []
                for i in range(self.in_channels):
                    pred_trend.append(
                        self.Linear_Trend[i](x_trend[:, :, i]))
                pred_trend = torch.stack(pred_trend, dim=2)
            else:  # same model for all channels
                pred_trend = self.Linear_Trend(x_trend)
        else:   
            x_s = x

        xf = self.stft.transform(x_s)
        
        ### Latent Prediction ###
        if self.independent_freqs:
            xf_pred = []
            
            # Loop through each frequency component
            for f in range(self.n_freqs):
                if self.individual:
                    # Use separate model for each channel at this frequency
                    xf_pred.append([
                        self.Linear_Freq_Latent[f][i](xf[:, i, f, :])
                        for i in range(2 * self.in_channels)
                    ])
                else:
                    # Use shared model for this frequency component
                    xf_pred.append(self.Linear_Freq_Latent[f](xf[:, :, f, :]))
            
            # Stack the predictions along the channels
            if self.individual:
                xf_pred = [torch.stack(preds, dim=1) for preds in xf_pred]
            
            # stack along frequency dimension
            xf_pred = torch.stack(xf_pred, dim=2)
        
        else:
            # Use a single model for all frequencies
            if self.individual:
                xf_pred = [
                    self.Linear_Freq_Latent[i](xf[:, i, :, :])
                    for i in range(2 * self.in_channels)]
                xf_pred = torch.stack(xf_pred, dim=1)
            else:
                # Use one shared model for all frequencies and channels
                xf_pred = self.Linear_Freq_Latent(xf.flatten(0, 1))

                xf_pred = xf_pred.view(
                    xf.shape[0], xf.shape[1], xf.shape[2], self.latent_pred_len)

        x_pred = self.stft.inverse_transform(xf_pred)  # (b, l, c)

        if self.isolate_trend:
            x_pred = x_pred + pred_trend

        return x_pred

    def _get_latent_length(self) -> Tuple[int, int]:
        """
        Compute the latent length for both the input sequence
        and the prediction horizon.
        
        Returns:
            tuple: (latent_length_input, latent_length_pred)
        """
        batch_size = 1
        fake_data = torch.randn(batch_size, self.seq_len, self.in_channels)
        
        # Compute the time-frequency representation using time_to_timefreq
        time_freq_data = self.stft.transform(fake_data)
        
        fake_data = torch.randn(batch_size, self.pred_len, self.in_channels)

        latent_length_input = time_freq_data.size(3)

        time_freq_data = self.stft.transform(fake_data)
        
        latent_length_pred = time_freq_data.size(3)
        # print(f'latent shape of prediction horizon {time_freq_data.shape}')
        
        return latent_length_input, latent_length_pred
