from einops import rearrange
from exp.exp_pred_latent import Exp_Latent_Pred

import torch
from typing import List, Tuple


class Exp_VAE2D_Pred(Exp_Latent_Pred):
    """
    Experiment class for latent prediction
    with VAE2D preprocessing.
    """

    def _process_latents(self, segments: List[torch.Tensor]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode segments into latents (means and log variances)
        suitable for time-series prediction. (i.e., compressing all
        channels and latent dimensions into one)

        Parameters
        ----------
        segments : List[torch.Tensor]
            List of input segments,
            each of shape (batch_size, channels, seq_len).

        Returns
        -------
        torch.Tensor
            Latents prepared for prediction models,
            shape (batch_size, total_length, latent_dim),
            where the last axis includes
            all the means and log variances,
            and total_length is the length of
            latents.

        """
        lf_latents = []
        hf_latents = []
        with torch.no_grad():
            for seg in segments:
                mu_l, log_var_l, mu_h, log_var_h = self.pretrained_model.encode(seg)
                 
                if self.pretrained_model.latent_type == 'spatial':
                    # Reshape latents: (batch_size, h, w, latent_dim)
                    # -> (batch_size, w, h * latent_dim)
                    mu_l = rearrange(mu_l, 'b h w d -> b w (h d)')
                    log_var_l = rearrange(log_var_l, 'b h w d -> b w (h d)')
                    mu_h = rearrange(mu_h, 'b h w d -> b w (h d)')
                    log_var_h = rearrange(log_var_h, 'b h w d -> b w (h d)')
                elif self.pretrained_model.latent_type == 'time':
                    # Latents are already in the desired shape
                    # (batch_size, w, latent_dim)
                    pass
                else:
                    raise ValueError(
                        f"Unknown latent_type: {self.pretrained_model.latent_type}",
                        level='error'
                    )

                # Concatenate means and log variances along the feature dimension
                lf_latent_segment = torch.cat([mu_l, log_var_l], dim=-1)
                hf_latent_segment = torch.cat([mu_h, log_var_h], dim=-1)

                # Collect LF and HF latent segments
                lf_latents.append(lf_latent_segment)
                hf_latents.append(hf_latent_segment)

        # Concatenate segments along the temporal dimension
        lf_latents = torch.cat(lf_latents, dim=1)
        hf_latents = torch.cat(hf_latents, dim=1)
        return lf_latents, hf_latents

    def _reconstruct_sequence(self, z_l: torch.Tensor,
                              z_h: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted latents back into the original sequence.

        Parameters
        ----------
        z_l, z_h : torch.Tensor
            Predicted latent representations,
            shape (batch_size, total_length, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed sequence of shape (batch_size, total_length, channels).
        """
        segments_l, segments_h = self._latent_to_segments(z_l, z_h)

        reconstructed_segments = []

        for seg_l, seg_h in zip(segments_l, segments_h):
            # Split into low-frequency (LF) and high-frequency (HF) components
            latent_dim_l = seg_l.size(-1) // 2  # latent dimensions (mu_l, log_var_l)
            latent_dim_h = seg_h.size(-1) // 2
            mu_l, log_var_l = seg_l[:, :, :latent_dim_l], seg_l[:, :, latent_dim_l:]
            mu_h, log_var_h = seg_h[:, :, :latent_dim_h], seg_h[:, :, latent_dim_h:]

            z_l = self.pretrained_model.reparameterize(mu_l, log_var_l)
            z_h = self.pretrained_model.reparameterize(mu_h, log_var_h)

            # Rearrange back to (batch_size, hid_dim, h, w) for decoding
            d = self.pretrained_model.hid_dim
            if self.pretrained_model.latent_type == 'spatial':
                z_l = rearrange(z_l, 'b w (h d) -> b h w d', d=d)
                z_h = rearrange(z_h, 'b w (h d) -> b h w d', d=d)
                z_l = self.pretrained_model.project_l(z_l)
                z_h = self.pretrained_model.project_h(z_h)
                z_l = rearrange(z_l, 'b h w d -> b d h w')
                z_h = rearrange(z_h, 'b h w d -> b d h w')
            elif self.pretrained_model.latent_type == 'time':
                z_l = self.pretrained_model.project_l(z_l)
                z_h = self.pretrained_model.project_h(z_h)
                z_l = rearrange(z_l, 'b w (d h) -> b d h w', d=d)
                z_h = rearrange(z_h, 'b w (d h) -> b d h w', d=d)

            reconstructed_segment = self.pretrained_model.decode(z_l, z_h)
            reconstructed_segments.append(reconstructed_segment)

        return torch.cat(reconstructed_segments, dim=1)
    