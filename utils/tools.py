import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List
import os
import yaml
import argparse
from copy import deepcopy

from utils.logger import Logger

plt.switch_backend('agg')


def adjust_learning_rate(optimizer: torch.optim.Optimizer,
                         epoch: int, args):
    """
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be adjusted.
    epoch : int
        The current epoch of training.
    args : object
        should have the following attributes:
        - `learning_rate` : float
            The initial learning rate for the optimizer.
        - `lradj` : str
            The learning rate adjustment strategy. Options include:
            - `'type1'` : Halve the learning rate every epoch.
            - `'type2'` : Use a predefined learning rate schedule:
                - Epoch 2-3: `5e-5`
                - Epoch 4-5: `1e-5`
                - Epoch 6-7: `5e-6`
                - Epoch 8-9: `1e-6`
                - Epoch 10-14: `5e-7`
                - Epoch 15-19: `1e-7`
                - after Epoch 20: `5e-8`
            - `'3'` : Constant learning rate for the first 10 epochs,
              then reduce to 10%.
            - `'4'` : Constant learning rate for the first 15 epochs,
              then reduce to 10%.
            - `'5'` : Constant learning rate for the first 25 epochs,
              then reduce to 10%.
            - `'6'` : Constant learning rate for the first 5 epochs,
              then reduce to 10%.
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    else:
        raise Exception(f'lradj {args.lradj} not identified, must be one of '
                        '[type1, type2, 3, 4, 5, 6]')

    # change the learning rate of optimizer
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.,
                 model_labels=None, logger: Optional[Logger]=None):
        """
        EarlyStopping class to stop training early
        based on validation loss.
        
        Parameters:
        - patience (int): How many epochs to wait after the last improvement.
        - verbose (bool): If True, prints the progress.
        - delta (float): Minimum change to qualify as an improvement.
        - model_labels (list or str or None): Optional list of labels for models.
          If provided, model filenames will use these labels
          instead of the default index.
          For single model, must accept a string
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_labels = model_labels
        self.logger = logger

    def __call__(self, val_loss, models, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            message = 'EarlyStopping counter: ' + \
                f'{self.counter} out of {self.patience}'

            if self.logger:
                self.logger.log(message, level='debug')
            else:
                print(message)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, path):
        if self.verbose and self.logger:
            self.logger.log(
                f'Validation loss decreased ({self.val_loss_min:.6f}'
                f' --> {val_loss:.6f}).  Saving model(s) ...',
                level='debug')
        elif self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f}'
                f' --> {val_loss:.6f}).  Saving model(s) ...'
            )

        if not isinstance(models, list):  # single model
            model_label = self.model_labels if (
                self.model_labels is not None) else f'checkpoint'
            checkpoint_path = f"{path}/{model_label}.pth"

            torch.save(models.state_dict(), checkpoint_path)
        else:  # save multiple models
            if self.model_labels is not None:
                if len(self.model_labels) != len(models):
                    raise ValueError(
                        f"The length of model_labels ({len(self.model_labels)})"
                        f" does not match the number of models ({len(models)}).")

            for idx, model in enumerate(models):
                model_label = self.model_labels[idx] if (
                    self.model_labels is not None) else f'checkpoint_{idx}'
                checkpoint_path = f"{path}/{model_label}.pth"

                torch.save(model.state_dict(), checkpoint_path)

        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true: np.ndarray, preds: Optional[np.ndarray]=None,
           title: Optional[str]=None,
           file_path: str='test.pdf'):
    """
    plot a time-series,
    with predicted values if provided.
    
    Parameters
    ----------
    true : np.ndarray
        The ground truth time-series data to be plotted.\n
        Should have shape (n_samples,).
        
    preds : np.ndarray, optional
        The predicted time-series data to be plotted. \n
        Should have shape (n_samples,). \n
        If not provided, only the ground truth will be plotted.

    title : str, optional
        If given, this will be used as plot title.
        Otherwise, no title for the plot.
        
    name : str, optional
        The file path where the plot will be saved. \n
        Should include file extension (e.g., '.pdf').\n
        Default is './pic/test.pdf'.
    """
    plt.figure()
    if title:
        plt.title(title)
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(file_path, bbox_inches='tight')


def visualise_results(folder_path, i, batch_x: torch.Tensor,
                      pred: np.ndarray, true: np.ndarray,
                      channel_idx: Optional[int]=None,
                      file_name: Optional[str]=None):
    """
    Visualise true time series vs prediction for the
    first sample of the batch.

    Parameters
    ----------
    i : int
        the batch index
    batch_x : torch.Tensor
        the input series
    pred, true : np.ndarray
        the predicted and ground truth values.
    channel_idx : int, optional
        if given, the channel will be plotted. \n
        Otherwise, a random channel is selected
    file_name : str, optional
        The file name (including postfix)
        to save the figure. e.g. figure.pdf. \n
        Default is pred_i.pdf where i is the
        batch id.
    """
    input = batch_x.detach().cpu().numpy()
    num_channels = input.shape[2]  # Number of channels
    if channel_idx is None:
        channel_idx = np.random.choice(num_channels)
    # concatenate batches
    gt = np.concatenate((input[0, :, channel_idx], true[0, :, channel_idx]), axis=0)
    pd = np.concatenate((input[0, :, channel_idx], pred[0, :, channel_idx]), axis=0)
    if file_name is None:
        file_name = 'pred_' + str(i) + '.pdf'
    visual(gt, pd,
           file_path=os.path.join(folder_path, file_name),
           title=f'Prediction of channel {channel_idx}')


def test_params_flop(model: torch.nn.Module, x_shape) -> None:
    """
    Compute FLOP of the model and print the computed complexities.

    Notes
    -----
    If you want to test former's FLOP,
    you need to give default value to inputs in model.forward(),
    the following code can only pass one argument to forward()

    Must install ptflops before use.
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def split_args_two_stages(args: argparse.Namespace):
    """
    Splits the namespace into two separate namespaces for
    reconstruction and prediction stages.

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments.

    Returns
    -------
    recon_args : argparse.Namespace
        Arguments for reconstruction.
    pred_args : argparse.Namespace
        Arguments for prediction.
    """
    # Convert to dictionary
    args_dict = vars(args)

    recon_args = deepcopy(args_dict)
    pred_args = deepcopy(args_dict)

    # Update stage-specific arguments
    recon_args["model"] = args.model_recon
    pred_args["model"] = args.model_pred
    recon_args["batch_size"] = args.batch_size_recon
    pred_args["batch_size"] = args.batch_size_pred
    recon_args["train_epochs"] = args.train_epochs_recon
    pred_args["train_epochs"] = args.train_epochs_pred
    recon_args["hop_length"] = args.hop_length_recon
    pred_args["hop_length"] = args.hop_length_pred
    recon_args["patience"] = args.patience_recon
    pred_args["patience"] = args.patience_pred
    recon_args["learning_rate"] = args.learning_rate_recon
    pred_args["learning_rate"] = args.learning_rate_pred

    # Convert back to namespaces
    recon_args = argparse.Namespace(**recon_args)
    pred_args = argparse.Namespace(**pred_args)

    return recon_args, pred_args


def plot_reconstruction_for_channels(
        origin: np.ndarray, recon: np.ndarray,
        channels_to_plot: List[int],
        save_path: Optional[str]=None,
        title: Optional[str]=None):
    """
    Plots the original and reconstructed time-series sequences
    for selected channels, and saves the figure.

    Parameters
    ----------
    origin, recon : numpy.ndarray
        The original and reconstructed time-series data with shape
        (time_steps, channels).

    channels_to_plot : list of int
        A list of channel indices to plot. \n

    save_path : str, optional
        The path where the generated plot will be saved.
        If not given, the canvas will be returned,
        but nothing saved. Please use plt.show()
        or plt.savefig() outside in this case.
    
    title : str, optional
        The overall title of the plot.
        Default is "Reconstruction of the Full Series"

    Returns
    -------
    None
        If save_path is given.
    fig, axes
        the canvas of matplotlib
        with the reconstruction figure
    """
    # Plot original vs reconstructed for selected channels
    fig, axes = plt.subplots(
        len(channels_to_plot), 1,
        figsize=(10, len(channels_to_plot) * 5))

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle('Reconstruction of the Full Series')

    if len(channels_to_plot) == 1:
        axes = [axes]

    for i, channel in enumerate(channels_to_plot):
        # Plot original sequence
        axes[i].plot(origin[:, channel], label="Original", alpha=0.7)
        axes[i].plot(recon[:, channel], label="Reconstructed", alpha=0.7)
        axes[i].set_title(f"Channel {channel}")
        axes[i].legend()

    # Save the figure to the specified path
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        return fig, axes


def format_large_int(n: int) -> str:
    """
    Format large integers into a readable string.
    Only support thousand(K), million (M) and billion (B).
    """
    if n >= 1e9:
        return f"{n / 1e9:,.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:,.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:,.2f}K"
    else:
        return f"{n:,.0f}"
    

def compute_kl_loss(mu: torch.Tensor,
                    log_var: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL prior loss for VAE model.

    Parameters
    ----------
    mu, log_var : torch.Tensor
        mean and log(variance),
        the two components of reparameterisation      
    
    Return
    ------
    torch.Tensor
        a scalar representing the KL loss.
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
