from einops import rearrange
from exp.exp_pred_latent import Exp_Latent_Pred

import torch
from typing import List, Tuple


class Exp_AE2D_Pred(Exp_Latent_Pred):
    """
    Experiment class for latent prediction
    with AE2D preprocessing.
    """
    def _process_latents(self, segments: List[torch.Tensor]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode segments into LF and HF latents using AE2D encoder.
        
        Parameters
        ----------
        segments : List[torch.Tensor]
            List of input segments, each of shape
            (batch_size, seq_len, channels).
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            LF and HF latent embeddings prepared for prediction models.
            Each tensor has shape (batch_size, temporal_length, latent_dim).
        """
        lf_latents = []
        hf_latents = []

        with torch.no_grad():
            for seg in segments:
                seg = rearrange(seg, 'b l c -> b c l')
                # Separate LF and HF components using AE2D encoder
                z_l = self.pretrained_model.encoder_l(seg)
                z_h = self.pretrained_model.encoder_h(seg)
                
                # Merge frequency and channel axes
                z_l = rearrange(z_l, 'b c h w -> b w (c h)')
                self.latent_h_l = z_l.shape[2]
                self.latent_c_l = z_l.shape[1]
                z_h = rearrange(z_h, 'b c h w -> b w (c h)')
                self.latent_h_h = z_h.shape[2]
                self.latent_c_h = z_h.shape[1]

                lf_latents.append(z_l)
                hf_latents.append(z_h)

        # Concatenate latents along the temporal dimension
        lf_latents = torch.cat(lf_latents, dim=1)
        hf_latents = torch.cat(hf_latents, dim=1)

        return lf_latents, hf_latents

    def _reconstruct_sequence(self, z_l: torch.Tensor,
                              z_h: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted LF and HF latents back
        into the original sequence.
        
        Parameters
        ----------
        z_l, z_h : torch.Tensor
            Predicted LF and HF latent representations,
            each of shape (batch_size, temporal_length, latent_dim).
        
        Returns
        -------
        torch.Tensor
            Reconstructed sequence of shape
            (batch_size, total_length, channels).
        """
        segments_l, segments_h = self._latent_to_segments(z_l, z_h)

        reconstructed_segments = []

        for seg_l, seg_h in zip(segments_l, segments_h):
            c = self.pretrained_model.hid_dim
            seg_l = rearrange(seg_l, 'b w (c h) -> b c h w', c=c)
            seg_h = rearrange(seg_h, 'b w (c h) -> b c h w', c=c)

            # Decode LF and HF components separately
            xhat_l = self.pretrained_model.decoder_l(seg_l)
            xhat_h = self.pretrained_model.decoder_h(seg_h)

            # Sum LF and HF components to get the reconstructed segment
            xhat = xhat_l + xhat_h
            reconstructed_segment = rearrange(xhat, 'b c l -> b l c')
            reconstructed_segments.append(reconstructed_segment)

        # Concatenate all reconstructed segments along the temporal dimension
        return torch.cat(reconstructed_segments, dim=1)
