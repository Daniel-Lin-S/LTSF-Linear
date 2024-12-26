from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate

import torch
import torch.nn as nn
from torch import optim

import os
import time
import numpy as np


class Exp_Recon(Exp_Basic):
    """
    Train and test a reconstruction model.
    """

    def __init__(self, args):
        """
        Parameter
        ---------
        args: object
            must contain the following attributes
            - model (str): the name of the reconstruction model (e.g., VAE, VQVAE).
            - use_multi_gpu, use_gpu (bool): if both true, set up GPU for running the codes.
            - device_ids (list of int or torch.device): CUDA devices.
            - train_epochs (int): number of training epochs.
            - learning_rate (float): the initial learning rate of the model.
            - patience (int): number of steps for early stopping.
            - features (str): strategy of handling channels ('M', 'S', 'MS').
            - train_only (bool): if True, train on the whole dataset (no validation and test).
            - checkpoints (str): path to save the checkpoints.
            - lradj (str): learning rate adjustment mode.
            - test_flop (bool): if True, compute the floating-point operations.
            - attributes for dataset and data loader (see data_provider).
        """
        super(Exp_Recon, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'VAE': VAE2D,  # Placeholder: Add the VAE model class.
            'VQVAE': VQVAE  # Placeholder: Add the VQVAE model class.
        }
        if self.args.model not in model_dict:
            raise ValueError(
                f"Invalid model name '{self.args.model}'. Available models are: "
                + ", ".join(model_dict.keys())
            )
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if not self.args.data.startswith('recon'):
            raise ValueError(
                "data must points to a reconstruction dataset "
                "starting with 'recon'."
            )
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self) -> callable:
        """
        Define criterion used to evaluate reconstruction.

        Return
        ------
        callable
            A function that takes two inputs of shape (batch_size, seq_len, channels)
            and returns a reconstruction loss.
        """
        criterion = nn.MSELoss()
        return criterion

    def _compute_recon_loss(self, criterion: callable, batch_x, outputs):
        """
        Evaluate reconstruction loss using a criterion.

        Parameters
        ----------
        criterion : function
            Must take two tensors and return a score(s).
        batch_x, outputs : tensor
            Input and reconstructed values both of shape (batch_size, seq_len, channels).
        """
        loss = criterion(outputs, batch_x)
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = self._compute_recon_loss(criterion, batch_x, outputs)
                else:
                    outputs = self.model(batch_x)
                    loss = self._compute_recon_loss(criterion, batch_x, outputs)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = self._compute_recon_loss(criterion, batch_x, outputs)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = self._compute_recon_loss(criterion, batch_x, outputs)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self._print_training_loss(time_now, train_steps, epoch, iter_count, i, loss)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
