import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time
from math import gcd
from typing import List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from inspect import signature

from data_provider.data_factory import data_provider
from utils.tools import (
    EarlyStopping, adjust_learning_rate, plot_reconstruction_for_channels
)
from utils.logger import Logger
from models.reconstructors import VAE2D, AE2D
from exp.exp_basic import Exp_Basic


class Exp_Recon(Exp_Basic):
    """
    Experiment class for training and testing 
    a reconstruction model.

    Notes
    -----
    - The model must be a torch.nn.Module
      with forward method taking batch_x, batch_id
      and other necessary arguments,
      which can be set up in the _configure_epoch_args function.
    - The forward method must return a dictionary
      of the losses.
    - For visualisation of a batch while validating,
      define the forward method with
      save_recon (bool) and folder_path (str)
      arguments. Otherwise, no batch-wise test figure will be
      plotted. The folder_path points to test_results folder.
    """
    def __init__(self, args, config, logger: Logger):
        """
        Initialize the VAE2D experiment.

        Parameters
        ----------
        args : object
            Contains all configuration parameters including:
            - model: Name of the model (string)
            - seq_len: Input sequence length
            - enc_in: Number of input channels
            - train_epochs: Number of training epochs
            - learning_rate: Learning rate for optimizer
            - patience: Early stopping patience
            - checkpoints: Path to save checkpoints
        config : dict
            Configuration file for initializing the model.
        logger : DualLogger
            the logger used to store messages into log file
            and print messages into console.
        """
        self.config = config
        super(Exp_Recon, self).__init__(args, logger)

    def _build_model(self) -> nn.Module:
        """
        Build the model.

        Returns
        -------
        torch.nn.Module
            Initialised model.

        Notes
        -----
        model initalisation must take two arguments:
        args, config. Where args is a Namespace
        with attributes recon_len (input length)
        and enc_in (number of input channels)
        and other training parameters.
        config is a dictionary taken from
        a yaml file consisting of the
        model-specific arguments.
        """
        model_dict = {
            'VAE' : VAE2D,
            'AE' : AE2D
        }

        args = self.args
        if args.model not in model_dict:
            self.logger.log(
                f"Invalid model name '{args.model}'. Available models are: "
                + ", ".join(model_dict.keys()),
                level='error'
            )
            raise

        recon_len = gcd(self.args.seq_len, self.args.pred_len)
        args.recon_len = recon_len
        self.logger.log(f'Reconstruction length: {recon_len}', level='debug')

        model = model_dict[args.model](args, self.config)
        self._print_model(model, self.args.model)

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
        dataset, dataloader = data_provider(
            self.args, flag, mode='recon',
            logger=self.logger)
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

    def _configure_epoch_args(self, epoch: int):
        """
        Configure model-specific arguments
        for each training epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch.

        Returns
        -------
        args : dict
            A dictionary of arguments to pass to the model.
        """
        # Default args, can be expanded based on model type
        epoch_args = {}

        if isinstance(self.model, VAE2D):
            init_beta = self.config['vae']['beta_init']
            final_beta = self.config['vae']['beta']
            beta = init_beta + (final_beta - init_beta) * (
                epoch / self.args.train_epochs)
            epoch_args['beta'] = beta
        
        return epoch_args

    def train(self, setting: str,
              log_interval: int=50):
        """
        Train the model.

        Parameters
        ----------
        setting : str
            A unique identifier for the training run.
        log_interval : int
            Number of batches between which the
            batch loss should be printed.

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
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True,
            model_labels='reconstructor', logger=self.logger)

        model_optim, scheduler = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_args = self._configure_epoch_args(epoch)
            epoch_time = time.time()

            # build progress bar
            train_bar = tqdm(train_loader,
                             desc=f"Epoch {epoch+1} [Training]",
                             leave=False)
            for i, (batch_x, _) in enumerate(train_bar):
                batch_x = batch_x.float().to(self.device)

                model_optim.zero_grad()
                loss_dict = self.model(batch_x, i, **epoch_args)
                try:
                    loss = loss_dict['total']
                except KeyError:
                    self.logger.log(
                        'KeyError: The loss dictionary must include a key "total", '
                        'indicating the overall loss. '
                        'Please ensure model.forward returns '
                        'a properly formatted loss dictionary.',
                        level="error"
                    )
                    raise

                if torch.isnan(loss):
                    self.logger.log(
                        'Loss becomes NaN while training, stopping the process...',
                        level='error'
                    )
                    raise

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                train_bar.set_postfix(average_loss=np.mean(train_loss))

                iter_count += 1
                if (i + 1) % log_interval == 0:
                    batch_message = f"Epoch: {epoch+1}, " + \
                        f"Step: {i+1}/{train_steps}, " + \
                            f"Total Loss: {loss.item():.4f}; "
                    detailed_message = "Loss breakdown: " + \
                        f"{', '.join([f'{key}: {value:.4f}'
                                      for key, value in loss_dict.items()
                                      if key != 'total'])}"
                    self.logger.log(batch_message+detailed_message,
                                    level='debug')

            epoch_duration = time.time() - epoch_time
            train_loss = np.average(train_loss)

            vali_loss = self.vali(
                vali_data, vali_loader, setting, epoch, flag='Vali')
            test_loss = self.vali(
                test_data, test_loader, setting, epoch, flag='Test')
            test_time = time.time() - epoch_duration - epoch_time

            message = f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, " + \
                f"Vali Loss: {vali_loss:.4f}, Test Loss: {test_loss:.4f}" + \
                    f", Training Time: {epoch_duration:.2f}s, " + \
                        f"Testing Time {test_time:.2f}s"

            self.logger.log(message)
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                self.logger.log("Early stopping ...")
                break

            scheduler.step()
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            for param_group in model_optim.param_groups:
                lr_message = f"Learning Rate updated to {param_group['lr']}"
                self.logger.log(lr_message, level='debug')

        # restore the model with lowest validation loss
        self.logger.log('Retrieving the best model ...', level='debug')
        best_model_path = os.path.join(path, 'reconstructor.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        test_loss = self.vali(
                test_data, test_loader, setting, epoch, plot_recon=True,
                flag='Final test')
        self.logger.log(f'Final test loss: {test_loss}')
        
        folder_path = './test_results/' + setting
        full_data = np.concatenate((train_data.data, vali_data.data, test_data.data),
                                   axis=0)  # join temporal axis
        lengths = [train_data.data.shape[0], vali_data.data.shape[0],
                   test_data.data.shape[0]]
        self.plot_reconstruction(full_data, folder_path, lengths)

        return self.model

    def vali(self, vali_data, vali_loader, setting, epoch_id,
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
        epoch_args = self._configure_epoch_args(epoch_id)

        folder_path = './test_results/' + setting
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_params = signature(self.model.forward).parameters
        save_recon_possible = 'save_recon' in model_params and (
            'folder_path' in model_params)
        if plot_recon and not save_recon_possible:
            plot_recon = False

        with torch.no_grad():
            vali_bar = tqdm(vali_loader,
                            desc=f"Epoch {epoch_id+1} [Validation]",
                            leave=False)
            for i, (batch_x, _) in enumerate(vali_bar):
                batch_x = batch_x.float().to(self.device)

                if i % 20 == 0 and plot_recon:
                    save_recon = True
                else:
                    save_recon = False

                loss_dict = self.model(
                    batch_x, i, save_recon=save_recon,
                    folder_path=folder_path, **epoch_args)
                
                total_loss.append(loss_dict['total'].item())

            loss_message = f"{flag} Loss breakdown: " + \
                f"{', '.join([f'{key}: {value:.4f}'
                              for key, value in loss_dict.items()
                              if key != 'total'])}"
            self.logger.log(loss_message, level='debug')

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
        If length of data is not divisible by self.args.recon_len,
        the last section will be cut off and ignored.
        """
        self.model.eval()

        if data.shape[1] < 3:
            channel_idx = list(range(data.shape[1]))

        segments = []
        for start_idx in range(0, data.data.shape[0], self.args.recon_len):
            end_idx = start_idx + self.args.recon_len
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
        self.logger.log(f'saved reconstruction figures to {folder_path}',
                        level='debug')
        plt.close(fig)
