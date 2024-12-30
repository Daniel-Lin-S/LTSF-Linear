import torch
from torch import nn
import torch.nn.functional as F
import os
from einops import rearrange
from typing import Tuple, Optional

from layers.VQVAE_EncDec import VQVAEEncoder, VQVAEDecoder
from utils.time_freq import (
    compute_downsample_rate,
    zero_pad_low_freq, zero_pad_high_freq,
    plot_lfhf_reconstruction,
    stft_lfhf
)


# TODO - add an option to turn of the LF-HF separation

class VAE2D(nn.Module):
    """
    Unsupervised learning stage using VAE with
    LF-HF separation and STFT.
    """
    def __init__(self, args,
                 config: dict):
        super().__init__()
        # input shape: (batch_size, input_length, in_channels)
        self.in_channels = args.enc_in
        self.input_length = args.recon_len
        self.config = config

        self.n_fft = config['n_fft']
        h = round(self.n_fft / 2) + 1  # number of frequency bins
        self.h = h
        init_dim = config['encoder']['init_dim']
        hid_dim = config['encoder']['hid_dim']
        self.hid_dim = hid_dim
        latent_dim = config['vae']['latent_dim']  # VAE latent dimension
        self.latent_type = config['vae']['latent_type']
        downsampled_width_l = config['encoder']['downsampled_width']['lf']
        downsampled_width_h = config['encoder']['downsampled_width']['hf']
        downsampled_height = config['encoder']['downsampled_width']['freq']
        downsampling = config['encoder']['downsampling']
        if downsampling == 'time':
            downsample_rate_l = compute_downsample_rate(
                self.input_length, self.n_fft, downsampled_width_l,
                downsampling)
            downsample_rate_h = compute_downsample_rate(
                self.input_length, self.n_fft, downsampled_width_h,
                downsampling)
        elif downsampling == 'freq':
            downsample_rate_l, h = compute_downsample_rate(
                self.input_length, self.n_fft, downsampled_height,
                downsampling, return_width=True)
            downsample_rate_h = downsample_rate_l

        # encoder
        self.encoder_l = VQVAEEncoder(
            init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
            config['encoder']['n_resnet_blocks'], zero_pad_high_freq, self.n_fft,
            config['encoder']['frequency_bandwidth'], downsampling)
        self.encoder_h = VQVAEEncoder(
            init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
            config['encoder']['n_resnet_blocks'], zero_pad_low_freq, self.n_fft,
            config['encoder']['frequency_bandwidth'], downsampling)

        # total dimensions of encoded component at each time stamp
        if self.latent_type == 'time':
            stamp_dim = hid_dim * h
        elif self.latent_type == 'spatial':
            stamp_dim = hid_dim
        else:
            raise NotImplementedError(
                "only supports VAE latent type ['time', 'spatial']"
                ""
            )

        ### Latent layers for VAE ###
        self.fc_mu_l = torch.nn.Linear(stamp_dim, latent_dim)
        self.fc_logvar_l = torch.nn.Linear(stamp_dim, latent_dim)
        self.fc_mu_h = torch.nn.Linear(stamp_dim, latent_dim)
        self.fc_logvar_h = torch.nn.Linear(stamp_dim, latent_dim)

        if stamp_dim != hid_dim:
            self.project_l = torch.nn.Linear(latent_dim, stamp_dim)
            self.project_h = torch.nn.Linear(latent_dim, stamp_dim)
        else:
            self.project_l = torch.nn.Identity()
            self.project_h = torch.nn.Identity()

        # Decoder
        self.decoder_l = VQVAEDecoder(
            init_dim, hid_dim, 2 * self.in_channels,
            downsample_rate_l, config['decoder']['n_resnet_blocks'],
            self.input_length, zero_pad_high_freq, self.n_fft, self.in_channels,
            config['encoder']['frequency_bandwidth'], downsampling)
        self.decoder_h = VQVAEDecoder(
            init_dim, hid_dim, 2 * self.in_channels,
            downsample_rate_h, config['decoder']['n_resnet_blocks'],
            self.input_length, zero_pad_low_freq, self.n_fft, self.in_channels,
            config['encoder']['frequency_bandwidth'], downsampling)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick.
        
        Mean mu and log variance log_var can
        have any shape, but must be consistent.
        """
        std = torch.exp(0.5 * log_var)

        if torch.isinf(std).any():
            raise ValueError(
                "Standard deviation contains Inf values (overflow), "
                " VAE forwarding stopped")
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _check_input(self, batch_x: torch.Tensor) -> None:
        if torch.isnan(batch_x).any():
            raise ValueError(
                'Input cannot have nan values'
            )
        if batch_x.ndim != 3 or (
            batch_x.shape[1] != self.input_length) or (
            batch_x.shape[2] != self.in_channels
        ):
            raise AssertionError(
                'input shape does not match with the settings. '
                f'Expected: (batch_size, {self.input_length}, {self.in_channels})'
                f', actual: {batch_x.shape}'
            )

    def forward(self, batch_x: torch.Tensor, batch_idx: int,
                beta: Optional[float]=None,
                return_x_rec: bool = False,
                save_recon: bool=False,
                folder_path: Optional[str]=None):
        """
        :param beta: a value between 0 and 1, the
          weight on regularisation power of KL loss
        :param return_x_rec: if True, directly return the
          reconstructed series x_rec.
        :param save_recon: if True, the reconstructions
          of LF and HF components are plotted and saved
          into a pdf files.
        :param folder_path: path to the folder in which
          the reconstruction figures will be plotted.
          Only used if save_recon=True

        Notes
        -----
        input and output are both of shape
        (batch_size, length, channels)
        """
        if beta is None: # use default value
            beta = self.config['vae']['beta']
        
        self._check_input(batch_x)

        # input shape (batch_size, length, channels)
        x = rearrange(batch_x, 'b l c -> b c l')

        # separate low and high-frequency components
        x_l, x_h = stft_lfhf(x, self.n_fft)

        ### Convolution Encoding ###
        z_l = self.encoder_l(x)
        z_h = self.encoder_h(x)

        ### VAE reparameterisation and sampling ###
        z_l_repar, mu_l, log_var_l = self._vae_process(
            z_l, self.fc_mu_l, self.fc_logvar_l, self.project_l)
        z_h_repar, mu_h, log_var_h = self._vae_process(
            z_h, self.fc_mu_h, self.fc_logvar_h, self.project_h)

        ### Decoder ###
        xhat_l = self.decoder_l(z_l_repar)
        xhat_h = self.decoder_h(z_h_repar)

        if return_x_rec:
            x_rec = xhat_l + xhat_h
            return rearrange(x_rec, 'b c l -> b l c')

        ### Compute Losses ###
        kl_loss_l = -0.5 * torch.sum(
            1 + log_var_l - mu_l.pow(2) - log_var_l.exp())
        kl_loss_h = -0.5 * torch.sum(
            1 + log_var_h - mu_h.pow(2) - log_var_h.exp())

        kl_loss = beta * (kl_loss_l + kl_loss_h)
        recon_loss_l = F.mse_loss(x_l, xhat_l)
        recon_loss_h = F.l1_loss(x_h, xhat_h)

        loss_dict = {
            'KL': kl_loss,
            'recon.LF.time': recon_loss_l,
            'recon.HF.time': recon_loss_h,
            'total': recon_loss_l + recon_loss_h + kl_loss
        }

        ### plot `x` and `xhat` (reconstruction) ###
        if save_recon:
            file_path = os.path.join(
                folder_path, 'recon_' + str(batch_idx) + '.pdf')
            plot_lfhf_reconstruction(
                x_l, xhat_l, x_h, xhat_h, file_path=file_path)

        return loss_dict

    def _vae_process(self, z: torch.Tensor, 
                     fc_mu: torch.nn.Module,
                     fc_logvar: torch.nn.Module,
                     project: torch.nn.Module) -> Tuple[
                         torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process encoding for a given encoder,
        compute the VAE latent variables.
        
        Parameters
        ----------
        z : torch.Tensor
            The embedded vector
        fc_mu : torch.nn.Module
            Linear layer to compute the mean (mu).
        fc_logvar : torch.nn.Module
            Linear layer to compute the log variance (log_var).
        project : torch.nn.Module
            Projection layer to apply after reparameterization.
            
        Returns
        -------
        z_sample : torch.Tensor
            The processed latent variable (z_sample)
            sampled from posterior distribtion
        mu, log_var : torch.Tensor
            the mean and log variance.
        """
        z_reshaped = self._reshape_latent(z)

        # Compute the mean and log variance
        mu, log_var = fc_mu(z_reshaped), fc_logvar(z_reshaped)

        # draw samples and reshape back to original dimensions
        z_sample = self.reparameterize(mu, log_var)
        z_sample = project(z_sample)

        if self.latent_type == 'time':
            C = self.hid_dim
            H = z_sample.shape[2] // C
            z_sample = rearrange(z_sample, 'b w (c h) -> b c h w',
                                 c=C, h=H)
        elif self.latent_type == 'spatial':
            z_sample = rearrange(z_sample, 'b h w c -> b c h w')

        return z_sample, mu, log_var
    
    def _reshape_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_type == 'time':
            # assign a distribution for each time stamp
            z_reshaped = rearrange(z, 'b c h w -> b w (c h)')
        elif self.latent_type == 'spatial':
            # assign a distribution for each spatial point
            z_reshaped = rearrange(z, 'b c h w -> b h w c')
        
        return z_reshaped

    def encode(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes the input time series into latent representations.
        (mean and log variances)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, length, channels).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            mu_l, log_var_l, mu_h, log_var_h
            means and log variances of LF and
            HF components respectively.
            Of shape (batch_size, h, w,
            latent_dim) if latent_type = 'spatial'
            (batch_size, w, latent_dim)
            if latent_type = 'time'.
        """
        x = rearrange(x, 'b l c -> b c l')
        z_l = self.encoder_l(x)
        z_h = self.encoder_h(x)

        z_l = self._reshape_latent(z_l)
        z_h = self._reshape_latent(z_h)

        mu_l, log_var_l = self.fc_mu_l(z_l), self.fc_logvar_l(z_l)
        mu_h, log_var_h = self.fc_mu_h(z_h), self.fc_logvar_h(z_h)

        return mu_l, log_var_l, mu_h, log_var_h

    def decode(self, z_l: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent representations back into the original space.

        Parameters
        ----------
        z_l, z_h : torch.Tensor
            Latent low-frequency and
            high-frequency representations.
            Of shape (batch_size, channels,
            h, w)

        Returns
        -------
        torch.Tensor
            Reconstructed time series of shape
            (batch_size, length, channels)
        """
        xhat_l = self.decoder_l(z_l)
        xhat_h = self.decoder_h(z_h)
        return rearrange(
            xhat_l + xhat_h, 'b c l -> b l c')
