from einops import rearrange
from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from models.VAE2D import VAE2D
from utils.metrics import metric
from utils.tools import visualise_results, EarlyStopping, adjust_learning_rate

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from math import gcd
import copy
import time

import os
from typing import List, Tuple, Optional


# TODO - now only supports linear prediction models, should add handling of transformers later
# TODO - allow for using different prediction models on LF and HF latents


class Exp_VAE2D_Pred(Exp_Main):
    """
    Experiment class for latent prediction with VAE preprocessing.
    """
    def __init__(self, args, 
                 model_path: str,
                 vae_config: dict):
        """
        Parameters
        ----------
        args : argparse.Namespace
            Experiment arguments.
        model_path : str
            File path to the pth file containing
            the pre-trained VAE2D model.
        vae_config: dict
            the configurations used when
            training the VAE2D modle
        """
        self.args = args
        self._load_vae_model(model_path, vae_config)
        self.device = self._acquire_device()

        shape_xl, shape_xh, shape_yl, shape_yh = self._get_latent_shapes()

        # independent prediction models for LF and HF
        model_args_l = copy.deepcopy(args)
        model_args_l.seq_len = shape_xl[1]
        model_args_l.pred_len = shape_yl[1]
        model_args_l.enc_in = shape_xl[2]
        print(f'number of latent channels {model_args_l.enc_in}')
        self.model_l = self._build_model(model_args_l).to(self.device)

        model_args_h = copy.deepcopy(args)
        model_args_h.seq_len = shape_xh[1]
        model_args_h.pred_len = shape_yh[1]
        model_args_h.enc_in = shape_xh[2]
        self.model_h = self._build_model(model_args_h).to(self.device)

    def _load_vae_model(self, model_path, vae_config) -> None:
        recon_len = gcd(self.args.seq_len, self.args.pred_len)
        self.recon_len = recon_len
        self.pred_segments = self.args.pred_len //recon_len
        self.pretrained_vae = VAE2D(
            in_channels=self.args.enc_in,
            input_length=recon_len,
            config=vae_config
        )

        try:
            self.pretrained_vae.load_state_dict(
                torch.load(model_path))
        except FileNotFoundError:
            raise FileNotFoundError(
                'Cannot find VAE checkpoint, please train the VAE model before '
                'running prediction experiment.')

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.pretrained_vae = nn.DataParallel(
                self.pretrained_vae, device_ids=self.args.device_ids)
        
        self.pretrained_vae.eval()
    
    def _get_latent_shapes(self):
        _, data_loader = self._get_data('train')

        for _, (batch_x, batch_y, _, _) in enumerate(data_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            x_segments = self._segment_sequence(batch_x, self.recon_len)
            y_segments = self._segment_sequence(batch_y, self.recon_len)
            x_latents_l, x_latents_h = self._process_latents(x_segments)
            y_latents_l, y_latents_h = self._process_latents(y_segments)

            break
        
        return (
            x_latents_l.shape, x_latents_h.shape,
            y_latents_l.shape, y_latents_h.shape)

    def _build_model(self, model_args):
        """
        Build the prediction model dynamically based on args.
        """
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        if model_args.pred_model not in model_dict:
            raise ValueError(
                f"Invalid model name '{model_args.pred_model}'. "
                "Available models are: "
                + ", ".join(model_dict.keys())
            )
        model = model_dict[model_args.pred_model].Model(model_args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        total_params = sum(p.numel() 
                           for p in model.parameters() 
                           if p.requires_grad)
        print('Number of trainable parameters of the '
              f'prediction model {model_args.pred_model}: {total_params}')
        return model

    def _segment_sequence(self, x: torch.Tensor,
                          seq_len: int) -> List[torch.Tensor]:
        """
        Segment a sequence into smaller subsequences.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, length, channels).
        seq_len : int
            Desired segment length.

        Returns
        -------
        List[torch.Tensor]
            List of segmented tensors.
        """
        if x.shape[1] % seq_len != 0:
            raise ValueError(
                f'temporal length of x, {x.shape[1]} '
                f'must be divisible by seq_len {seq_len}')
        segments = [x[:, i:i+seq_len, :] for i in range(0, x.size(1), seq_len)]
        return segments

    def _process_latents(self, segments: List[torch.Tensor]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode segments into latents suitable for time-series prediction.

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
                mu_l, log_var_l, mu_h, log_var_h = self.pretrained_vae.encode(seg)
                 
                if self.pretrained_vae.latent_type == 'spatial':
                    # Reshape latents: (batch_size, h, w, latent_dim)
                    # -> (batch_size, w, h * latent_dim)
                    mu_l = rearrange(mu_l, 'b h w d -> b w (h d)')
                    log_var_l = rearrange(log_var_l, 'b h w d -> b w (h d)')
                    mu_h = rearrange(mu_h, 'b h w d -> b w (h d)')
                    log_var_h = rearrange(log_var_h, 'b h w d -> b w (h d)')
                elif self.pretrained_vae.latent_type == 'time':
                    # Latents are already in the desired shape
                    # (batch_size, w, latent_dim)
                    pass
                else:
                    raise NotImplementedError(
                        f"Unknown latent_type: {self.pretrained_vae.latent_type}")

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
        seq_len_l = z_l.shape[1] // self.pred_segments
        seq_len_h = z_h.shape[1] // self.pred_segments
        # Split the latents back into segments
        segments_l = torch.split(z_l, seq_len_l, dim=1)
        segments_h = torch.split(z_h, seq_len_h, dim=1)

        reconstructed_segments = []

        with torch.no_grad():
            for seg_l, seg_h in zip(segments_l, segments_h):
                # Split into low-frequency (LF) and high-frequency (HF) components
                latent_dim_l = seg_l.size(-1) // 2  # latent dimensions (mu_l, log_var_l)
                latent_dim_h = seg_h.size(-1) // 2
                mu_l, log_var_l = seg_l[:, :, :latent_dim_l], seg_l[:, :, latent_dim_l:]
                mu_h, log_var_h = seg_h[:, :, :latent_dim_h], seg_h[:, :, latent_dim_h:]

                z_l = self.pretrained_vae.reparameterize(mu_l, log_var_l)
                z_h = self.pretrained_vae.reparameterize(mu_h, log_var_h)

                # Rearrange back to (batch_size, hid_dim, h, w) for decoding
                d, h = self.pretrained_vae.hid_dim, self.pretrained_vae.h
                if self.pretrained_vae.latent_type == 'spatial':
                    z_l = rearrange(z_l, 'b w (h d) -> b h w d', h=h, d=d)
                    z_h = rearrange(z_h, 'b w (h d) -> b h w d', h=h, d=d)
                    z_l = self.pretrained_vae.project_l(z_l)
                    z_h = self.pretrained_vae.project_h(z_h)
                    z_l = rearrange(z_l, 'b h w d -> b d h w')
                    z_h = rearrange(z_h, 'b h w d -> b d h w')
                elif self.pretrained_vae.latent_type == 'time':
                    z_l = self.pretrained_vae.project_l(z_l)
                    z_h = self.pretrained_vae.project_h(z_h)
                    z_l = rearrange(z_l, 'b w (d h) -> b d h w', h=h, d=d)
                    z_h = rearrange(z_h, 'b w (d h) -> b d h w', h=h, d=d)

                reconstructed_segment = self.pretrained_vae.decode(z_l, z_h)
                reconstructed_segments.append(reconstructed_segment)

        return torch.cat(reconstructed_segments, dim=1)
    
    def _predict_batch(self,
                       batch_x: torch.Tensor,
                       batch_y: Optional[torch.Tensor]=None,
                       criterion: Optional[callable]=None,
                       calculate_loss: bool=True):
        batch_x = batch_x.float().to(self.device)

        x_segments = self._segment_sequence(batch_x, self.recon_len)
        x_latents_l, x_latents_h = self._process_latents(x_segments)

        # Train latent prediction
        pred_latents_l = self.model_l(x_latents_l)  # (batch_size, length, channels)
        pred_latents_h = self.model_h(x_latents_h)

        if calculate_loss:
            batch_y = batch_y.float().to(self.device)
            y_segments = self._segment_sequence(batch_y, self.recon_len)
            y_latents_l, y_latents_h = self._process_latents(y_segments)
            loss_l = criterion(pred_latents_l, y_latents_l)
            loss_h = criterion(pred_latents_h, y_latents_h)
            return loss_l,loss_h
        else:
            return pred_latents_l, pred_latents_h

    def _get_data(self, flag):
        """
        Load the dataset and preprocess it for latent prediction.
        """
        dataset, data_loader = data_provider(self.args, flag)
        return dataset, data_loader

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val') if not self.args.train_only else None

        model_optim_l = optim.Adam(self.model_l.parameters(), lr=self.args.learning_rate)
        model_optim_h = optim.Adam(self.model_h.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        model_labels = ['pred_lf', 'pred_hf']
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True, model_labels=model_labels)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        log_interval = 50  # Log every 'log_interval' batches

        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()
            self.model_l.train()
            self.model_h.train()
            train_loss_l = []
            train_loss_h = []
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                batch_start_time = time.time()

                # Segment, encode, and concatenate latents
                loss_l, loss_h = self._predict_batch(batch_x, batch_y, criterion)

                model_optim_l.zero_grad()
                loss_l.backward(retain_graph=True)  # Retain computation graph for HF backprop
                model_optim_l.step()
                train_loss_l.append(loss_l.item())

                model_optim_h.zero_grad()
                loss_h.backward()
                model_optim_h.step()
                train_loss_h.append(loss_h.item())

                if i % log_interval == 0:
                    print(f"Epoch [{epoch + 1}/{self.args.train_epochs}], "
                        f"Batch [{i}/{len(train_loader)}], "
                        f"Time taken per batch: {time.time()-batch_start_time:.2f}s, "
                        f"Average batch Loss LF: {np.mean(train_loss_l):.4f}, "
                        f"Average batch Loss HF: {np.mean(train_loss_h):.4f}")

            # Validation and epoch logging
            epoch_duration = time.time() - epoch_start_time
            if vali_loader is not None:
                vali_loss_l, vali_loss_h = self.vali(vali_loader, criterion)

                print(
                    f"Epoch {epoch + 1}/{self.args.train_epochs}, "
                    f"Time taken: {epoch_duration:.2f}s, "
                    f"Train Loss LF: {np.mean(train_loss_l):.4f}, "
                    f"Train Loss HF: {np.mean(train_loss_h):.4f}, "
                    f"Val Loss LF: {vali_loss_l:.4f}, "
                    f"Val Loss HF: {vali_loss_h:.4f}"
                )
                early_stopping(vali_loss_l+vali_loss_h,
                               [self.model_l, self.model_h], path)
            else:
                train_loss = np.mean(train_loss_l) + np.mean(train_loss_h)
                print(
                    f"Epoch {epoch + 1}/{self.args.train_epochs}, "
                    f"Time taken: {epoch_duration:.2f}s, "
                    f"Train Loss LF: {np.mean(train_loss_l):.4f}, "
                    f"Train Loss HF: {np.mean(train_loss_h):.4f}"
                )
                early_stopping(train_loss,
                               [self.model_l, self.model_h], path)

            adjust_learning_rate(model_optim_l, epoch + 1, self.args)
            adjust_learning_rate(model_optim_h, epoch + 1, self.args)

    def vali(self, vali_loader, criterion):
        self.model_l.eval()
        self.model_h.eval()

        vali_loss_l = []
        vali_loss_h = []

        with torch.no_grad():
            for _, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                # Segment, encode, and process latents
                loss_l, loss_h = self._predict_batch(batch_x, batch_y, criterion)

                vali_loss_l.append(loss_l.item())
                vali_loss_h.append(loss_h.item())

        return np.mean(vali_loss_l), np.mean(vali_loss_h)

    def test(self, setting, test=0):
        """
        Evaluate the model on the test set, report metrics, and save results.

        Parameters
        ----------
        setting : str
            Identifier for the experiment (e.g., model configuration).
        test : bool, optional
            If True, load the best checkpoint for evaluation. Default is False.
        """
        _, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading checkpoint models')
            self.model_l.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'pred_lf.pth')))
            self.model_h.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'pred_hf.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model_l.eval()
        self.model_h.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_latents_l, pred_latents_h = self._predict_batch(
                    batch_x, calculate_loss=False
                )

                pred_seq = self._reconstruct_sequence(
                    pred_latents_l, pred_latents_h)
                
                true_seq, pred_seq = self._extract_prediction(batch_y, pred_seq)

                pred = pred_seq.detach().cpu().numpy()
                true = true_seq.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:  # Visualise predictions every 20 batches
                    visualise_results(folder_path, i, batch_x, pred, true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        ### Save results ###
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        metrics = metric(preds, trues)
        print('mse:{}, mae:{}'.format(metrics['mse'], metrics['mae']))

        # Write metrics into a text file
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(
                metrics['mse'], metrics['mae'], metrics['rse'], metrics['corr']))
            f.write('\n\n')

        return metrics
