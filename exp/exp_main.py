from data_provider.data_factory import data_provider, DataLoader
from data_provider.data_loader import Dataset
from exp.exp_basic import Exp_Basic
from models import (
    Informer, Autoformer, Transformer,
    DLinear, Linear, NLinear, FDLinear,
    STFTLinear, PatchTST, FEDformer
)
from utils.tools import (
    EarlyStopping, adjust_learning_rate, visualise_results,
     test_params_flop, format_large_int
)
from utils.metrics import metric, decay_l2_loss
from utils.logger import Logger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from typing import Optional, Tuple
from tqdm import tqdm
import os
import time
import warnings
import csv


warnings.filterwarnings('ignore')

# list of available available
model_dict = {
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Informer': Informer,
    'DLinear': DLinear,
    'NLinear': NLinear,
    'Linear': Linear,
    'FDLinear': FDLinear,
    'STFTLinear': STFTLinear,
    'PatchTST': PatchTST,
    'FEDformer': FEDformer
}


class Exp_Main(Exp_Basic):
    """
    Train and test a prediction model.

    MSE used to 
    """
    def __init__(self, args, logger: Optional[Logger]=None):
        """
        Parameter
        ---------
        args: object
            must contain the following attributes
            - model (str): the name of the model used.
            - use_multi_gpu, use_gpu (bool): if both true,
              set up GPU for running the codes
            - device_ids (list of int or torch.device): CUDA devices
            - train_epochs (int): number of training epochs
            - learning_rate (float): the initial learning rate of the model
            - patience (int): number of steps after which
              EarlyStop takes place. Validation loss
              monitored criterion, unless train_only=True,
              in which case training loss is used.
            - seq_len (int): length of input sequence
            - label_len (int): length of overlap between
              input and prediction horizons
            - pred_len (int): length of the prediction horizon
            - output_attention (bool): if true, transformer-based models
              will return the attentions as well as outputs
            - use_amp (bool): if true, use automatic mixed-precision.
              i.e. some operations are performed at lower precision
              to save memory.
            - features (str): strategy of handling channels,
              must be one of ['M', 'S', 'MS']
              M - multivariate;
              S - univariate;
              MS - multivariate dataset, but only predict
              the LAST variable.
            - train_only (bool): if True, train on
              the whole dataset. (no validation and test)
            - checkpoints (str): path to save the checkpoints,
              e.g. './checkpoints/'
            - lradj (str): learning rate adjust mode
              see docstrings of utils.tools.adjust_learning_rate
            - test_flop (bool): if True, compute the floating
              point operations. See details in
              utils.tools.test_params_flop
            - attributes for dataset and data loader,
              see docstrings of data_provider

            May add attributes for
            - initialisation of the model.
            
        """
        super(Exp_Main, self).__init__(args, logger)

    def _build_model(self, model_args=None):
        args = model_args if model_args is not None else self.args

        if args.model not in model_dict:
            error_message = (
                f"Invalid model name '{args.model}'. "
                "Available models are: "
                + ", ".join(model_dict.keys()))
            
            raise ValueError(
                error_message
            )

        model = model_dict[args.model].Model(args).float()

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        self._print_model(model, self.args.model)

        return model

    def _get_data(self, flag) -> Tuple[Dataset, DataLoader]:
        data_set, data_loader = data_provider(
            self.args, flag, mode='pred', logger=self.logger)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _get_model_size(self) -> str:
        """ get the total number of trainable parameters """
        n_params = sum(p.numel()
                       for p in self.model.parameters()
                       if p.requires_grad)
        return format_large_int(n_params)

    def _get_loss(self) -> callable:
        """
        define criterion used to evaluate the prediction

        Return
        ------
        callable
            a function that takes two inputs
            of shape (batch_size, pred_len, channels)
            and return a score
        """
        available_losses = ['mse', 'l1', 'decay_mse']
        if self.args.loss == 'mse':
            return nn.MSELoss()
        elif self.args.loss == 'l1':
            return nn.L1Loss()
        elif self.args.loss == 'decay_mse':
            return decay_l2_loss
        else:
            raise NotImplementedError(
                f'Given loss type {self.args.loss}'
                'is not supported. Available options: '
                f"{', '.join(available_losses)}."
            )

    def _extract_prediction(
            self, batch_y, outputs
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slice out the prediction horizon and the
        variable (channel) for evaluation if necessary.
        """
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
        return batch_y, outputs

    def _compute_pred_loss(self, criterion: callable,
                           batch_y, outputs):
        """
        Evaluate the quality of prediction using a criterion

        Parameters
        ----------
        criterion : function
            must take two tensors and return a
            score(s)
        batch_y, outputs : tensor
            the true and predicted values
            both of shape (batch_size, length, channels)

        Notes
        -----
        Temporal length of both tensors can be longer than pred_len
        but only the last pred_len entries will be compared.
        """
        batch_y, outputs = self._extract_prediction(batch_y, outputs)
        loss = criterion(outputs, batch_y)
        return loss

    def _get_output(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """ Give the model required inputs """
        if 'Linear' in self.args.model:
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if outputs.shape[1] != self.args.pred_len:
            error_msg = (
                f'The temporal length of model output: {outputs.shape[1]}'
                f', does not match prediction length {self.args.pred_len}'
            )
            self.logger.log(error_msg, 'error')
            raise RuntimeError(
                error_msg
            )

        return outputs
    
    def _prepare_decoder_input(self, batch_y):
        """
        Create an initial input for decoder, consisting of
        first label_len time stamps of y and
        the prediction horizon is initialised to 0.
        This is used by many transformer-based prediction models.

        :param batch_y: tensor of shape (batch_size, length, channels)

        :return tensor: the decoder input of shape
        (batch_size, label_len + pred_len, channels)
        """
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp],
            dim=1).float().to(self.device)
        return dec_inp


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            vali_bar = tqdm(vali_loader,
                             desc=f"[Validating]",
                             leave=False)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark
                    ) in enumerate(vali_bar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._prepare_decoder_input(batch_y)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._get_output(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._get_output(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y, outputs = self._extract_prediction(batch_y, outputs)

                # remove from computational graph
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, logger=self.logger)

        model_optim = self._select_optimizer()
        criterion = self._get_loss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            train_bar = tqdm(train_loader,
                             desc=f"Epoch {epoch+1} [Training]",
                             leave=False)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark
                    ) in enumerate(train_bar):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = self._prepare_decoder_input(batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._get_output(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        loss = self._compute_pred_loss(criterion, batch_y, outputs)
                        train_loss.append(loss.item())
                else:
                    outputs = self._get_output(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    loss = self._compute_pred_loss(criterion, batch_y, outputs)
                    train_loss.append(loss.item())
                
                train_bar.set_postfix(average_loss=np.mean(train_loss))

                if (i + 1) % 100 == 0: # log every 100 steps
                    self._log_training_loss(
                        time_now, train_steps, epoch, iter_count, i, loss)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_msg = "Epoch: {0} cost time: {1:.4f}s".format(
                epoch + 1, time.time() - epoch_time)
            if self.logger:
                self.logger.log(epoch_msg)
            else:
                print(epoch_msg)

            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                msg = ("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                       "Vali Loss: {3:.7f} Test Loss: {4:.7f}").format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss)

                early_stopping(vali_loss, self.model, path)
            else:
                msg = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss)

                early_stopping(train_loss, self.model, path)
            
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)

            if early_stopping.early_stop:
                if self.logger:
                    self.logger.log("Early stopping...")
                else:
                    print("Early stopping...")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            for param_group in model_optim.param_groups:
                lr_message = f"Learning Rate updated to {param_group['lr']}"
                if self.logger:
                    self.logger.log(lr_message, level='debug')
                else:
                    print(lr_message)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _log_training_loss(self, prev_time: float, train_steps: int,
                             epoch: int, iter_count: int, i: int,
                             loss: torch.Tensor):
        """
        Print a line indicating training time, current loss and epoch number.

        Parameters
        ----------
        prev_time : float
            The previoius time stamp recorded
        train_steps : int
            Number of training batches in one epoch
        epoch : int
            The index of current epoch
        iter_count : int
            The index of current batch
        i : int
            The index of current batch
        loss : torch.Tensor
            The current training loss to be logged
        """
        msg1 = "\tEpoch: {0}, Batch: {1} | loss: {2:.7f}".format(
            epoch + 1, i + 1, loss.item())
        speed = (time.time() - prev_time) / iter_count
        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
        msg2 = '\tspeed: {:.4f}s/iter; Estimated left time: {:.4f}s'.format(
            speed, left_time)
        
        msg = msg1 + msg2
        if self.logger:
            self.logger.log(msg, level='debug')
        else:
            print(msg)

    def test(
            self, setting: str, model_id: str, exp_id: str,
            exp_seed: int, load: bool=False
        ) -> None:
        """
        Test the model and save the results.

        Parameters
        ----------
        setting : str
            The identifier of this experiment.
        model_id : str
            The identifier of this model,
            composing of model name and relevant hyperparameters.
        exp_id : str
            The identifier of this experiment,
            with information like dataset name, prediction length,
            input length etc.
        exp_seed : int
            The random seed used in the experiment.
        load : bool, optional
            if True, load the model from the checkpoint
            to test the model. \n
            Default is False.
        """
        _, test_loader = self._get_data(flag='test')
        
        if load:
            file_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            try:
                self.model.load_state_dict(
                    torch.load(file_path))
            except FileNotFoundError:
                error_msg = (
                    'Cannot find model checkpoint, please train the model'
                    ' before testing.'
                )
                raise FileNotFoundError(error_msg)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_bar = tqdm(test_loader,
                            desc=f"[Testing]",
                            leave=False)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark
                    ) in enumerate(test_bar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._prepare_decoder_input(batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._get_output(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._get_output(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark)

                batch_y, outputs = self._extract_prediction(batch_y, outputs)
                # print(outputs.shape,batch_y.shape)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:  # visualise predictions every 20 batches
                    visualise_results(folder_path, i, batch_x, pred, true)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        self._save_results(
            setting, model_id, exp_id, exp_seed, preds, trues, inputx,
            output_file=self.args.result_file)

        return

    def _save_results(self, setting: str, model_id: str,
                      exp_id: str, exp_seed: int,
                      preds: torch.Tensor, trues: torch.Tensor,
                      inputx: torch.Tensor,
                      output_file: str='results.csv',
                      save_all=False) -> None:
        """
        Calculate the metrics and save them to result.txt
        file, and print MSE, MAE.

        Parameters
        ----------
        setting : str
            The full experimenting settings used to name the folder
            in which the results will be saved when save_all=True
        model_id : str
            The identifier of this model,
            composing of model name and relevant hyperparameters.
        exp_id : str
            The identifier of this experiment,
            with information like dataset name, prediction length,
            input length etc.
        exp_seed : int
            The random seed used in the experiment.
        preds, trues : torch.Tensor
            the predicted values and ground truth,
            both of shape (batch_size, pred_len, channels)
        inputx : torch.Tensor
            the input series used to produce the
            predicted values.
            Shape: (batch_size, seq_len, channels)
        output_file : str
            Path to the txt file in which
            the metrics will be stored.
            Default is 'result.txt'
        save_all : bool, optional
            if True, save the preds, trues, inputx
            and all metrics as numpy value files.
            Warning: this can consume a lot of disk space
            if the dataset is large.
            Default is False.
        """
        # flatten batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        ### result save ###
        # logger
        metrics = metric(preds, trues)
        metric_msg = 'mse:{}, mae:{}'.format(metrics['mse'], metrics['mae'])
        if self.logger:
            self.logger.log(metric_msg)
        else:
            print(metric_msg)

        # write metrics into a csv file
        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # If the file is empty, write the header first
            if f.tell() == 0:
                writer.writerow([
                    'Experiment', 'Model', 'Seed', 'Model Size',
                    'MSE', 'MAE', 'RSE', 'MSPE', 'MAPE'])
            
            # Write the experiment data
            model_size = self._get_model_size()
            row = [
                exp_id,
                model_id,
                exp_seed,
                model_size,
                metrics['mse'],
                metrics['mae'],
                metrics['rse'],
                metrics['mspe'],
                metrics['mape']
            ]
            writer.writerow(row)

        self.logger.log('metrics saved to output_file', level='debug')

        if save_all:
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.savez(folder_path + 'metrics.npz', **metrics)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'x.npy', inputx)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            try:
                self.model.load_state_dict(torch.load(best_model_path))
            except FileNotFoundError:
                error_msg = (
                    'Cannot find model checkpoint, please train the model'
                    ' before running predict function.'
                )
                raise FileNotFoundError(error_msg)

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark
                    ) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._prepare_decoder_input(batch_y)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._get_output(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._get_output(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        ### save results ###
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(
            np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
            columns=pred_data.cols).to_csv(
                folder_path + 'real_prediction.csv', index=False)

        return
