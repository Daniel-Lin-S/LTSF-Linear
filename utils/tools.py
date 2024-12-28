import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional
import os
import yaml
import argparse
from copy import deepcopy

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
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.,
                 model_labels=None):
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

    def __call__(self, val_loss, models, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f}'
                f' --> {val_loss:.6f}).  Saving model ...')
        if not isinstance(models, list): # single model
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
           name='./pic/test.pdf'):
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
        
    name : str, optional
        The file path where the plot will be saved. \n
        Should include file extension (e.g., '.pdf').\n
        Default is './pic/test.pdf'.
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visualise_results(folder_path, i, batch_x, pred, true):
    """
    Visualise true time series vs prediction for the
    first sample of the batch.

    Notes
    -----
    Only the last channel (variable) will be plotted.
    """
    input = batch_x.detach().cpu().numpy()
    # concatenate batches
    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


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
