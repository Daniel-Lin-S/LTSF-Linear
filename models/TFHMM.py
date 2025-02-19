import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange

from layers.Decompositions import SeasonTrendDecomp
from utils.time_freq import STFT


class Model(nn.Module):
    """
    Use Hidden Markov Model (HMM) to predict the seasonal component
    in time-frequency domain,
    and linear model for the trend component.
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
            - `kernel_size` (int): Kernel size for the trend decomposition.
            - `n_states` (int): Number of hidden states in the HMM.
            - `hid_dim` (int): Hidden dimension for the state probability model.
            - `nfft` (int): Number of FFT points.
            - `stft_hop_length` (int): Hop length for the STFT transformation.
            - `individual` (bool): Whether to use separate
              linear models for each input channel.      
        """

        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.n_freqs = configs.nfft // 2 + 1
        self.individual = configs.individual
        self.n_states = configs.n_states
        self.hid_dim = configs.hid_dim

        self.requires_time_markers = False

        self.stft = STFT(configs.nfft, configs.stft_hop_length)

        self.latent_pred_len = self._get_latent_length(self.pred_len)

        self.emission_dim = 2 * self.in_channels * self.n_freqs

        # matrices for HMM
        self.trans_mat = nn.Parameter(
            torch.randn(self.n_states, self.n_states))
        self.emit_mat = nn.Parameter(
            torch.randn(self.n_states, 2*self.in_channels, self.n_freqs))

        # learn state probabilities from emissions using MLP
        self.state_prob_model = nn.Sequential(
            nn.Linear(self.emission_dim, configs.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(configs.hid_dim, self.n_states)
        )

        # linear model for trend component
        if self.individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.in_channels)])
        else:
            self.Linear_Trend = nn.Linear(
                self.seq_len, self.pred_len)

        self.decomp = SeasonTrendDecomp(
            kernel_size=configs.kernel_size)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, in_channels).
        
        Returns
        -------
        pred : torch.Tensor
            Predicted tensor of shape (batch_size, pred_len, in_channels).
        """

        x_season, x_trend = self.decomp(x)

        ### predict season ###
        xf = self.stft.transform(x_season)

        xf = rearrange(xf, 'b c f t -> b (c f) t')

        state_probs = self._compute_state_probs(xf)
        
        pred_state_probs = self.propagate_state_probs(state_probs)

        pred_emit = self.emit(pred_state_probs)  # (b, c, f, t)
        pred_season = self.stft.inverse_transform(pred_emit)

        #### predict trend ###
        if self.individual:
            pred_trend = []
            for i in range(self.in_channels):
                pred_trend.append(
                    self.Linear_Trend[i](x_trend[:, :, i]))
            pred_trend = torch.stack(pred_trend, dim=2)
        else:  # same model for all channels
            pred_trend = self.Linear_Trend(x_trend)

        return pred_season + pred_trend


    def _get_latent_length(self, seq_len: int) -> int:
        """
        Compute the latent length for both the input sequence
        and the prediction horizon.
        
        Returns
        -------
        latent_len : int
            The temporal length of the time-frequency graph
            obtained from sequence of length `seq_len`.
        """

        fake_data = torch.randn(1, seq_len, self.in_channels)
        xf = self.stft.transform(fake_data)
        latent_length = xf.size(3)
        
        return latent_length
    
    def _compute_state_probs(
            self, xf: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the state probabilities from the emissions.

        Parameters
        ----------
        xf : torch.Tensor
            The input emissions, shape (batch_size, emission_dim, len).
        
        Return
        ------
        state_probs : torch.Tensor
            The state probabilities for the input sequence,
            shape (batch_size, len, n_states).
        """
        if xf.ndim != 3:
            raise ValueError(
                "Input tensor should have 3 dimensions, "
                f"got shape {xf.shape}")

        len = xf.shape[2]

        xf = xf.view(-1, self.emission_dim) # (batch*len, emission_dim)
        state_probs = self.state_prob_model(xf)  # (batch*len, n_states)

        # normalise the state probabilities
        state_probs = F.softmax(state_probs, dim=-1)

        return state_probs.view(-1, len, self.n_states)

    
    def propagate_state_probs(self, state_probs: torch.Tensor):
        """
        Propagate the state probabilities into the
        future using the transition matrix.
        
        Parameters
        ----------
        state_probs : torch.Tensor
            The state probabilities for the input sequence,
            shape (batch_size, latent_input_len, n_states).
        n_times_pred : int
            The number of future time steps to predict.

        
        Returns:
        --------
        future_state_probs : torch.Tensor
            The predicted state probabilities for the future,
            shape (batch_size, latent_pred_len, n_states).
        """
        future_state_probs_list = []

        # Initialise the first time step with the last known state_probs
        future_state_probs_list.append(
            state_probs[:, -1, :].unsqueeze(1))  # (batch_size, 1, n_states)

        # Iteratively compute state probabilities
        for t in range(1, self.latent_pred_len + 1):
            next_step = torch.matmul(
                future_state_probs_list[-1],
                F.softmax(self.trans_mat, dim=-1)
            )

            future_state_probs_list.append(next_step)

        future_state_probs = torch.cat(future_state_probs_list, dim=1)
        
        return future_state_probs[:, 1:, :]

    def emit(self, state_probs: torch.Tensor):
        """
        Generate emission predictions from the future state probabilities.
        
        Parameters
        ----------
        state_probs : torch.Tensor
            The predicted state probabilities for the future,
            shape (batch_size, len, n_states).
        
        Returns
        -------
        emission_pred : torch.Tensor
            The predicted emissions for the future,
            shape (batch_size, 2*in_channels, n_freqs, len).
        """
        emission_pred = torch.zeros(
            state_probs.shape[0], 2*self.in_channels, self.n_freqs,
            state_probs.shape[1])
        
        emit_mat = self.emit_mat.unsqueeze(0).unsqueeze(0)  # (1, 1, n, c, f)
        state_probs = state_probs.unsqueeze(-1).unsqueeze(-1)  # (b, t, n, 1, 1)
        
        emission_pred = torch.sum(
            emit_mat * state_probs, dim=2)  # (b, t, c, f)
        
        return emission_pred.permute(0, 2, 3, 1)  # (b, c, f, t)