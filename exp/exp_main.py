from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visualise_results, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    """
    Train and test a prediction model.

    MSE used to 
    """
    def __init__(self, args):
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
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        if self.args.model not in model_dict:
            raise ValueError(
                f"Invalid model name '{self.args.model}'. Available models are: "
                + ", ".join(model_dict.keys())
            )
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self) -> callable:
        """
        define criterion used to evaluate the prediction

        Return
        ------
        callable
            a function that takes two inputs
            of shape (batch_size, pred_len, channels)
            and return a score
        """
        criterion = nn.MSELoss()
        return criterion

    def _extract_prediction(self, batch_y, outputs):
        """
        Slice out the prediction horizon and the
        variable (channel) for evaluation if necessary.
        """
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
        return batch_y,outputs

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
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return dec_inp


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
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

                    # print(outputs.shape,batch_y.shape)
                    loss = self._compute_pred_loss(criterion, batch_y, outputs)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0: # log every 100 steps
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
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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

    def _print_training_loss(self, time_now, train_steps: int,
                             epoch: int, iter_count: int, i: int, loss):
        """
        Print a line indicating training time, current loss and epoch number.
        """
        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        speed = (time.time() - time_now) / iter_count
        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._prepare_decoder_input(batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._get_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._get_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        ### result save ###
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._prepare_decoder_input(batch_y)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._get_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._get_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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
