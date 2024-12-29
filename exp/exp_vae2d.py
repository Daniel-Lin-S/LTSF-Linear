import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time
from math import gcd
from typing import List, Optional
import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, plot_reconstruction_for_channels
from models.VAE2D import VAE2D
from exp.exp_basic import Exp_Basic


class Exp_VAE2D(Exp_Basic):
    """
    Experiment class for training and testing the VAE2D model
    for reconstruction tasks.
    """
    def __init__(self, args, config):
        """
        Initialize the VAE2D experiment.

        Parameters
        ----------
        args : object
            Contains all configuration parameters including:
            - model: Name of the model (string)
            - seq_len: Input sequence length
            - enc_in: Number of input channels
            - n_fft: Number of FFT points
            - train_epochs: Number of training epochs
            - learning_rate: Learning rate for optimizer
            - patience: Early stopping patience
            - checkpoints: Path to save checkpoints
            - vae: Dictionary of VAE-specific hyperparameters
            - exp_params: Experiment parameters including beta for KL divergence
            - output_path: Path to save outputs
        config : dict
            Configuration file for initializing VAE2D model.
        """
        self.config = config
        self.init_beta = config['vae']['beta_init']
        self.final_beta = config['vae']['beta']
        super(Exp_VAE2D, self).__init__(args)

    def _build_model(self):
        """
        Build the VAE2D model.

        Returns
        -------
        VAE2D
            Initialised VAE2D model.
        """
        recon_len = gcd(self.args.seq_len, self.args.pred_len)
        self.recon_len = recon_len
        print(f'Reconstruction length: {recon_len}')
        model = VAE2D(
            in_channels=self.args.enc_in,
            input_length=recon_len,
            config=self.config
        )

        total_params = sum(p.numel() 
                           for p in model.parameters() 
                           if p.requires_grad)
        print('Number of trainable parameters of the '
              f'model {self.args.model}: {total_params}')

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(
                model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        """
        Get the dataset and dataloader.

        Parameters
        ----------
        flag : str
            Dataset split, e.g., 'train', 'val', 'test'.

        Returns
        -------
        dataset, DataLoader
        """
        dataset, dataloader = data_provider(self.args, flag, 'recon')
        return dataset, dataloader

    def _select_optimizer(self):
        """
        Select the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer, torch.optim.lr_scheduler
            The optimizer and learning rate scheduler
        """
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return optimizer, scheduler

    def adjusted_beta(self, epoch: int, total_epochs: int) -> float:
        """
        Gradual annealing schedule for the KL loss.
        Starts with a lower beta value and increases it gradually.

        Return
        ------
        float
            the beta-value for current epoch
        """
        return self.init_beta + (
            self.final_beta - self.init_beta) * (epoch / total_epochs)

    def train(self, setting):
        """
        Train the model.

        Parameters
        ----------
        setting : str
            A unique identifier for the training run.

        Returns
        -------
        Trained model.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, scheduler = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            beta = self.adjusted_beta(epoch, self.args.train_epochs)

            for i, batch in enumerate(train_loader):
                batch_x, _ = batch  # Assuming batch contains only data and no labels
                batch_x = batch_x.float().to(self.device)

                model_optim.zero_grad()
                recons_loss, kl_losses = self.model(batch_x, i, beta=beta)
                loss = recons_loss['LF.time'] + recons_loss['HF.time'] + kl_losses['combined']

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                iter_count += 1
                if (i + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1}, Step: {i+1}/{train_steps}, "
                          f"Total loss: {loss.item():.4f}; "
                          f"Reconstruction losses: LF {recons_loss['LF.time']:.4f}, "
                          f"HF {recons_loss['HF.time']:.4f}; "
                          f"KL loss: {kl_losses['combined']:.4f}")

            print(f"Epoch: {epoch+1} cost time: {(time.time() - epoch_time):.2f}s")
            train_loss = np.average(train_loss)

            vali_loss = self.validate(
                vali_data, vali_loader, setting, epoch, flag='Vali')
            test_loss = self.validate(
                test_data, test_loader, setting, epoch, flag='Test')

            print(
                f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}"
                f", Vali Loss: {vali_loss:.4f}, Test Loss: {test_loss:.4f}")
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            for param_group in model_optim.param_groups:
                print(f"Epoch {epoch+1} Learning Rate: {param_group['lr']}")

        # restore the model with lowest validation loss
        print('Training finished, saving the best model...')
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        test_loss = self.validate(
                test_data, test_loader, setting, epoch, plot_recon=True,
                flag='Final test')
        print(f'Final test loss: {test_loss}')
        
        folder_path = './test_results/' + setting
        full_data = np.concatenate((train_data.data, vali_data.data, test_data.data),
                                   axis=0)  # join temporal axis
        lengths = [train_data.data.shape[0], vali_data.data.shape[0],
                   test_data.data.shape[0]]
        self.plot_reconstruction(full_data, folder_path, lengths)

        return self.model

    def validate(self, vali_data, vali_loader, setting, epoch_id,
                 plot_recon: bool=False, flag='Vali'):
        """
        Validate the model.

        Parameters
        ----------
        vali_data : Dataset
        vali_loader : DataLoader
        setting : str
            A unique identifier for the training run.
        epoch_id : int
            index of current epoch
        plot_recon : bool, optional
            if True, the reconstructions will be saved into
            pdf files in the
            folder /test_results/reonstructions/

        Returns
        -------
        Validation loss (float).
        """
        self.model.eval()
        total_loss = []
        beta = self.adjusted_beta(epoch_id, self.args.train_epochs)

        folder_path = './test_results/' + setting
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            recon_losses_lf = []
            recon_losses_hf = []
            recon_losses_kl = []
            for i, batch in enumerate(vali_loader):
                batch_x, _ = batch
                batch_x = batch_x.float().to(self.device)

                if i % 20 == 0 and plot_recon:
                    save_recon = True
                else:
                    save_recon = False

                recons_loss, kl_losses = self.model(
                    batch_x, i, save_recon=save_recon,
                    folder_path=folder_path, beta=beta)
                recon_losses_lf.append(recons_loss['LF.time'].item())
                recon_losses_hf.append(recons_loss['HF.time'].item())
                recon_losses_kl.append(kl_losses['combined'].item())

                loss = recons_loss['LF.time'] + recons_loss['HF.time'] + kl_losses['combined']
                total_loss.append(loss.item())

            print(f"{flag} loss breakdown -- "
                  f"Reconstruction losses: LF {np.average(recon_losses_lf)}"
                  f", HF {np.average(recon_losses_hf)}; "
                  f"KL loss: {np.average(recon_losses_kl)}")

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def plot_reconstruction(self, data: np.ndarray,
                            folder_path: str,
                            lengths: Optional[List[int]]=None,
                            channel_idx: List[int]=[0, 1, 2],
                            ):
        """
        Parameters
        ----------
        data : numpy.ndarray
            2-dimensional array of shape
            (length, channels)
        folder_path : str
            the folder path to which the reconstruction
            figure should be saved.
        lengths : List[int], optional
            if given, the train-validation-test
            split points will be plotted.
        channel_idx : list of int, optional
            A list of channel indices to plot. \n
            Default is 0, 1, 2.

        Notes
        -----
        If length of data is not divisible by self.recon_len,
        the last section will be cut off and ignored.
        """
        self.model.eval()

        if data.shape[1] < 3:
            channel_idx = list(range(data.shape[1]))

        segments = []
        for start_idx in range(0, data.data.shape[0], self.recon_len):
            end_idx = start_idx + self.recon_len
            if end_idx < data.shape[0]:
                segments.append(data[start_idx : end_idx])

        reconstructed_data = []
        for segment in segments:
            batch_x = torch.Tensor(segment).unsqueeze(0)  # (1, recon_len, channels)
            batch_idx = torch.tensor([0])
            x_rec = self.model(batch_x, batch_idx, return_x_rec=True)
            reconstructed_data.append(x_rec.squeeze(0))

        recon = torch.cat(reconstructed_data, dim=0).cpu().detach().numpy()

        file_path = f'{folder_path}/recon_full.pdf'
        fig, axes = plot_reconstruction_for_channels(data, recon, channel_idx)

        for ax in axes:
            ax.axvline(x=lengths[0], color='red', linestyle='--',
                       label="Train-Val Split")
            ax.axvline(x=lengths[0] + lengths[1], color='green',
                       linestyle='--', label="Val-Test Split")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
