import torch
from torch import nn
import torch.nn.functional as F
import os
from einops import rearrange
from typing import Optional
from layers.VQVAE_EncDec import VQVAEEncoder, VQVAEDecoder
from utils.time_freq import (
    compute_downsample_rate,
    zero_pad_low_freq, zero_pad_high_freq,
    stft_lfhf
)
from utils.tools import plot_reconstruction_for_channels


class AE2D(nn.Module):
    """
    Simple autoencoder for time-frequency representation, using
    2D convolutional layers. \n
    Supports optional low-frequency and high-frequency separation.
    """
    def __init__(self, args, config: dict):
        super().__init__()
        self.in_channels = args.enc_in
        self.input_length = args.recon_len
        self.config = config

        self.n_fft = config['n_fft']
        self.separation = config['lfhf_separation']
        init_dim = config['encoder']['init_dim']
        hid_dim = config['encoder']['hid_dim']
        self.hid_dim = hid_dim
        downsampling = config['encoder']['downsampling']

        # Compute downsample rates based on separation mode
        frequency_bandwidth = config['encoder']['frequency_bandwidth']
        n_resnet_blocks_en = config['encoder']['n_resnet_blocks']
        n_resnet_blocks_de = config['decoder']['n_resnet_blocks']
        if self.separation:
            downsample_rate_l = compute_downsample_rate(
                self.input_length, self.n_fft,
                config['encoder']['downsampled_width']['lf'], downsampling
            )
            downsample_rate_h = compute_downsample_rate(
                self.input_length, self.n_fft,
                config['encoder']['downsampled_width']['hf'], downsampling
            )

            # Encoders and decoders for LF and HF components
            self.encoder_l = VQVAEEncoder(
                init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
                n_resnet_blocks_en, zero_pad_high_freq, self.n_fft,
                frequency_bandwidth, downsampling
            )
            self.encoder_h = VQVAEEncoder(
                init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
                n_resnet_blocks_en, zero_pad_low_freq, self.n_fft,
                frequency_bandwidth, downsampling
            )
            self.decoder_l = VQVAEDecoder(
                init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
                n_resnet_blocks_de, self.input_length, zero_pad_high_freq,
                self.n_fft, self.in_channels,
                frequency_bandwidth, downsampling
            )
            self.decoder_h = VQVAEDecoder(
                init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
                n_resnet_blocks_de, self.input_length, zero_pad_low_freq,
                self.n_fft, self.in_channels, frequency_bandwidth, downsampling
            )
        else:
            downsample_rate = compute_downsample_rate(
                self.input_length, self.n_fft,
                config['encoder']['downsampled_width']['combined'], downsampling
            )
            self.encoder = VQVAEEncoder(
                init_dim, hid_dim, 2 * self.in_channels, downsample_rate,
                n_resnet_blocks_en, None, self.n_fft,
                frequency_bandwidth, downsampling
            )
            self.decoder = VQVAEDecoder(
                init_dim, hid_dim, 2 * self.in_channels, downsample_rate,
                n_resnet_blocks_de, self.input_length, None,
                self.n_fft, self.in_channels, frequency_bandwidth, downsampling
            )

    def forward(self, batch_x: torch.Tensor, batch_idx: int,
                return_x_rec: bool = False, save_recon: bool = False,
                folder_path: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass for reconstruction.

        Parameters
        ----------
        batch_x : torch.Tensor
            Input tensor of shape (batch_size, length, channels).
        batch_idx : int
            Index of the current batch (used for plotting purposes).
        return_x_rec : bool, optional
            If True, directly return the reconstructed time series. \n
            plotting will be disabled.
        save_recon : bool, optional
            If True, save the reconstruction plot.
        folder_path : str, optional
            Path to save the reconstruction plot if save_recon=True.

        Returns
        -------
        torch.Tensor
            If return_x_rec=True, returns the reconstructed tensor
            with shape (batch_size, length, channels).
        """
        x = rearrange(batch_x, 'b l c -> b c l')

        if self.separation:
            # Separate LF and HF components
            x_l, x_h = stft_lfhf(x, self.n_fft)

            # Encode and decode LF and HF components
            z_l = self.encoder_l(x_l)
            z_h = self.encoder_h(x_h)
            xhat_l = self.decoder_l(z_l)
            xhat_h = self.decoder_h(z_h)

            x_rec = xhat_l + xhat_h

            if return_x_rec:
                return rearrange(x_rec, 'b c l -> b l c')

            recon_loss_l = F.mse_loss(x_l, xhat_l)
            recon_loss_h = F.l1_loss(x_h, xhat_h)
            loss_dict = {
                'recon.LF.time' : recon_loss_l,
                'recon.HF.time' : recon_loss_h,
                'total' : recon_loss_l + recon_loss_h
            }
        else:
            # No separation, single encoder-decoder pair
            z = self.encoder(x)
            x_rec = self.decoder(z)

            if return_x_rec:
                return rearrange(x_rec, 'b c l -> b l c')

            recon_loss = F.mse_loss(x, x_rec)

            loss_dict = {
                'recon.time' : recon_loss,
                'total' : recon_loss
            }

        if save_recon:
            file_path = os.path.join(folder_path, f'recon_{batch_idx}.pdf')
            x_rec = rearrange(x_rec, 'b c l -> b l c')
            x_true = batch_x.detach().cpu().numpy()
            x_rec = x_rec.detach().cpu().numpy()
            plot_reconstruction_for_channels(
                x_true[0], x_rec[0], [0, 1, 2], file_path,
                title=f'Reconstructions of Test Batch {batch_idx}'
            )

        return loss_dict
