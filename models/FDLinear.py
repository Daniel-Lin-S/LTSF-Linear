import torch
import torch.nn as nn
from layers.Decompositions import FreqDecomp


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
        self.individual = configs.individual

        self.requires_time_markers = False

        self.series_decomp = FreqDecomp(
            nfft=configs.nfft, hop_length=configs.stft_hop_length,
            n_low_freqs=max(int(configs.nfft / 8), 1),
            split_trend=True
        )  

        # Linear layers for frequency components
        if self.individual:   # Separate linear models for each channel
            # same linear model for LF and HF components
            self.Linear_Freq = nn.ModuleList()
            for _ in range(self.in_channels):
                self.Linear_Freq.append(nn.Linear(self.seq_len, self.pred_len))
            # linear model for trend component
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.in_channels):
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:  # Shared linear model for all channels
            self.Linear_Freq = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            

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
        x_l, x_h, x_trend = self.series_decomp(x)  # (b, l, c)

        if self.individual:  # Apply separate linear models to each channel
            x_l_pred = torch.zeros([x_l.shape[0], self.pred_len, x_l.shape[2]])
            x_h_pred = torch.zeros_like(x_l_pred)
            x_trend_pred = torch.zeros_like(x_l_pred)

            for i in range(self.in_channels):
                x_l_pred[:, :, i] = self.Linear_Freq[i](x_l[:, :, i])
                x_h_pred[:, :, i] = self.Linear_Freq[i](x_h[:, :, i])
                x_trend_pred[:, :, i] = self.Linear_Trend[i](x_trend[:, :, i])

        else:  # Apply the same linear model to all channels
            x_l_pred = self.Linear_Freq(x_l)
            x_h_pred = self.Linear_Freq(x_h)
            x_trend_pred = self.Linear_Trend(x_trend)

        # Combine low and high frequency components
        x_pred = x_l_pred + x_h_pred + x_trend_pred
        return x_pred
