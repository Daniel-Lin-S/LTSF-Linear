import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange
import os

from layers.VQVAE_EncDec import VQVAEEncoder, VQVAEDecoder
from layers.VQ import VectorQuantize
from utils.time_freq import (
    compute_downsample_rate,
    zero_pad_low_freq, zero_pad_high_freq,
    quantize, stft_decomp, plot_lfhf_reconstruction
)

class VQVAE(nn.Module):
    """
    VQVAE model with optional LF-HF separation.
    Returns either a loss dictionary or the reconstructed signal.
    """
    def __init__(self, args, config: dict):
        super().__init__()
        self.in_channels = args.enc_in
        self.input_length = args.recon_len
        self.config = config

        self.n_fft = config['n_fft']
        init_dim = config['encoder']['init_dim']
        hid_dim = config['encoder']['hid_dim']
        downsampled_width_l = config['encoder']['downsampled_width']['lf']
        downsampled_width_h = config['encoder']['downsampled_width']['hf']
        downsampling = config['encoder']['downsampling']

        downsample_rate_l = compute_downsample_rate(
            self.input_length, self.n_fft, downsampled_width_l, downsampling)
        downsample_rate_h = compute_downsample_rate(
            self.input_length, self.n_fft, downsampled_width_h, downsampling)

        # Encoders
        n_resnet_enc = config['encoder']['n_resnet_blocks']
        freq_bandwidth = config['encoder']['frequency_bandwidth']
        self.encoder_l = VQVAEEncoder(
            init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
            n_resnet_enc, zero_pad_high_freq, self.n_fft,
            freq_bandwidth, downsampling)
        self.encoder_h = VQVAEEncoder(
            init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
            n_resnet_enc, zero_pad_low_freq, self.n_fft,
            freq_bandwidth, downsampling)

        # Vector Quantizers
        codebook_params = config.get('VQ-VAE', {}).get('codebook', {})
        self.vq_model_l = VectorQuantize(
            hid_dim, config['VQ-VAE']['codebook_sizes']['lf'],
            codebook_params=codebook_params,
            **config['VQ-VAE'])
        self.vq_model_h = VectorQuantize(
            hid_dim, config['VQ-VAE']['codebook_sizes']['hf'],
            codebook_params=codebook_params,
            **config['VQ-VAE'])

        # Decoders
        n_resnet_dec = config['decoder']['n_resnet_blocks']
        self.decoder_l = VQVAEDecoder(
            init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
            n_resnet_dec, self.input_length, zero_pad_high_freq,
            self.n_fft, self.in_channels, freq_bandwidth,
            downsampling)
        self.decoder_h = VQVAEDecoder(
            init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
            n_resnet_dec, self.input_length, zero_pad_low_freq,
            self.n_fft, self.in_channels, freq_bandwidth,
            downsampling)

    def forward(self, batch_x: torch.Tensor, batch_idx: int,
                return_x_rec: bool = False,
                save_recon: bool=False,
                folder_path: Optional[str]=None):
        """
        Forward pass for VQVAE model.

        Parameters
        ----------
        batch_x : torch.Tensor
            Input time series tensor of shape
            (batch_size, length, channels).
        return_x_rec : bool, optional
            If True, returns only the reconstructed signal. \n
            If False, returns a loss dictionary.

        Returns
        -------
        Union[dict, torch.Tensor]
            If return_x_rec=False, returns a loss dictionary. \n
            If return_x_rec=True, returns the reconstructed signal,
            same shape as batch_x.
        """
        x = rearrange(batch_x, 'b l c -> b c l')

        # Split low and high-frequency components
        x_l, x_h = stft_decomp(x, self.n_fft)

        # Low-Frequency Path
        z_l = self.encoder_l(x)
        z_q_l, _, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        xhat_l = self.decoder_l(z_q_l)

        # High-Frequency Path
        z_h = self.encoder_h(x)
        z_q_h, _, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        xhat_h = self.decoder_h(z_q_h)

        # Reconstructed signal
        x_rec = xhat_l + xhat_h

        if return_x_rec:
            return rearrange(x_rec, 'b c l -> b l c')

        # Prepare loss dictionary
        loss_recon_lf = F.mse_loss(x_l, xhat_l)
        loss_recon_hf = F.l1_loss(x_h, xhat_h)

        loss = loss_recon_lf + loss_recon_hf + (
            vq_loss_l['loss'] + vq_loss_h['loss'])
        loss_dict = {
            'recon.LF.time': loss_recon_lf,
            'recon.HF.time': loss_recon_hf,
            'vq.LF': vq_loss_l['loss'],
            'vq.HF': vq_loss_h['loss'],
            'vq.commit.LF': torch.tensor(vq_loss_l['commit_loss']),
            'vq.orthogonal.LF': torch.tensor(vq_loss_l['orthogonal_reg_loss']),
            'vq.commit.HF': torch.tensor(vq_loss_h['commit_loss']),
            'vq.orthogonal.HF': torch.tensor(vq_loss_h['orthogonal_reg_loss']),
            'perplexity.LF': perplexity_l,
            'perplexity.HF': perplexity_h,
            'total': loss
        }

        if save_recon:
            file_path = os.path.join(
                folder_path, 'recon_' + str(batch_idx) + '.pdf')
            plot_lfhf_reconstruction(
                x_l, xhat_l, x_h, xhat_h, file_path=file_path)

        return loss_dict
