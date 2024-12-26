import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time
from math import gcd

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
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
        if not self.args.data.startswith('recon'):
            raise ValueError(
                "data must points to a reconstruction dataset "
                "starting with 'recon'."
            )
        dataset, dataloader = data_provider(self.args, flag)
        return dataset, dataloader

    def _select_optimizer(self):
        """
        Select the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
        """
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

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

        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                batch_x, _ = batch  # Assuming batch contains only data and no labels
                batch_x = batch_x.float().to(self.device)

                model_optim.zero_grad()
                recons_loss, kl_losses = self.model(batch_x, i)
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

            vali_loss = self.validate(vali_data, vali_loader, setting)
            test_loss = self.validate(test_data, test_loader, setting)

            print(
                f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}"
                f", Vali Loss: {vali_loss:.4f}, Test Loss: {test_loss:.4f}")
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # restore the model with lowest validation loss
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        test_loss = self.validate(
                test_data, test_loader, setting, plot_recon=True)
        return self.model

    def validate(self, vali_data, vali_loader, setting,
                 plot_recon: bool=False):
        """
        Validate the model.

        Parameters
        ----------
        vali_data : Dataset
        vali_loader : DataLoader
        setting : str
            A unique identifier for the training run.
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

        folder_path = './test_results/' + setting
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, _ = batch
                batch_x = batch_x.float().to(self.device)

                if i % 20 == 0 and plot_recon:
                    save_recon = True
                else:
                    save_recon = False

                recons_loss, kl_losses = self.model(
                    batch_x, i, save_recon=save_recon,
                    folder_path=folder_path)
                loss = recons_loss['LF.time'] + recons_loss['HF.time'] + kl_losses['combined']
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
