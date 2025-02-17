import argparse
import torch
import random
import numpy as np
import pandas as pd
import os
import sys

from utils.tools import load_yaml_param_settings
from utils.logger import DualLogger
from utils.configs import (
    split_args_two_stages, get_base_settings,
    get_pred_model_settings, get_recon_model_settings
)
from exp.exp_recon import Exp_Recon
from exp.exp_pred_vae import Exp_VAE2D_Pred
from exp.exp_pred_ae import Exp_AE2D_Pred


### Set global seed for reproducibility ###
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

### Collect Arguments ###
parser = argparse.ArgumentParser(description='Two-stage Prediction')

# basic config
parser.add_argument('--is_training', type=int, required=True, help='0 : no training,'
                    '1 - train prediction model only; 2 - train both reconstruction '
                    'and prediction model; 3 - train reconstruction only')
parser.add_argument('--exp_id', type=str, required=True, help='model id')
parser.add_argument('--model_recon', type=str, required=True,
                    help='name of reconstruction model, options: [VAE, AE]')
parser.add_argument('--model_pred', type=str, required=True,
                    help='name of prediction model, options: '
                    '[Autoformer, Informer, Transformer, DLinear, Linear, NLinear]')
parser.add_argument('--train_only', type=bool, required=False, default=False,
                    help='perform training on full input dataset without validation and testing')
parser.add_argument('--itr', type=int, default=2, help='number of experiment repetitions')
parser.add_argument('--test_idx', type=int, default=0,
                    help='index of the experiment repetition used for testing'
                    ' if is_training = 0')
parser.add_argument('--des', type=str, default='',
                    help='experiment description added at the end of folder name')
parser.add_argument('--result_file', type=str, default='result.csv',
                    help='file path of the csv file to store the evaluation results')
parser.add_argument('--log_file', type=str, default='logs/test.log',
                    help='file path of log file for the training process')
parser.add_argument('--rerun', action='store_true', default=False,
                    help='whether to cover existing test results')

# data loader
parser.add_argument('--data', type=str, required=True,
                    help='dataset type, Options: [custom, ETTh1, ETTh2, ETTm1, ETTm2]')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,'
                     ' S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, '
                    't:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]'
                    ', you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='location of model checkpoints')

# DLinear
parser.add_argument('--individual', action='store_true', default=False,
                    help='Linear, DLinear, NLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value embedding + temporal embedding + positional embedding '
                    '2: value embedding + temporal embedding 3: value embedding + positional embedding '
                    '4: value embedding')
# For linear models, this hyperparameter is the number of input channels
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') 
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96,
                    help='input sequence length') # 96=24*4
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--hop_length_pred', type=int, default=1,
                    help='hop length for the sliding window of getting dataset '
                    'of the prediction model')
parser.add_argument('--hop_length_recon', type=int, default=4,
                    help='hop length for the sliding window of getting dataset '
                    'of the reconstruction model')

# reconstruction model
parser.add_argument('--config', type=str, default='configs/config_vae.yaml',
                    help='file path of configuration file for the reconstruction model')

# optimization and training
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs_recon', type=int, default=10,
                    help='number of epochs for training reconstruction model')
parser.add_argument('--train_epochs_pred', type=int, default=10,
                    help='number of epochs for training prediction model')
parser.add_argument('--batch_size_recon', type=int, default=16,
                    help='batch size of reconstruction input data')
parser.add_argument('--batch_size_pred', type=int, default=32,
                    help='batch size of prediction input data')
parser.add_argument('--patience_recon', type=int, default=3,
                    help='early stopping patience of reconstruction model')
parser.add_argument('--patience_pred', type=int, default=3,
                    help='early stopping patience of prediction model')
parser.add_argument('--learning_rate_recon', type=float, default=0.001,
                    help='initial optimizer learning rate for reconstruction model')
parser.add_argument('--learning_rate_pred', type=float, default=0.005,
                    help='initial optimizer learning rate for prediction model')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--loss_level', type=str, default='origin',
                    help='One of latent and origin. Latent: prediction model trained at latent level'
                    '; origin: prediction model trained after decoding.')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id.')
parser.add_argument('--use_multi_gpu', action='store_true',
                    help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',
                    help='device ids of multile gpus')

args = parser.parse_args()

### Prepare Logger ###
logger = DualLogger(args.log_file)

logger.start_experiment(args.exp_id, args.is_training)
logger.log(f'Experiment settings: \n {args}', level='debug')

iteration_seeds = [random.randint(0, 2**32 - 1) for _ in range(args.itr)]

### Set up GPU devices ###
if not torch.cuda.is_available() and args.use_gpu:
    logger.log('use_gpu = True but GPU not available, '
               'deviced changed to cpu',
               level='warning')
    args.use_gpu = False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

### Record basic and model settings ###
config = load_yaml_param_settings(args.config)

base_setting = get_base_settings(args)
recon_model_setting = get_recon_model_settings(args, config)
pred_model_setting = get_pred_model_settings(args)
model_setting = f'{pred_model_setting}_{recon_model_setting}'
model_id = (f'{args.model_recon}_{recon_model_setting}_'
                 f'{args.model_pred}_{pred_model_setting}')

### Set specific latent prediction experiment ###
if args.model_recon == 'AE':
    Exp_pred = Exp_AE2D_Pred
elif args.model_recon == 'VAE':
    Exp_pred = Exp_VAE2D_Pred

### Two-stage Training and Testing ###
args_recon, args_pred = split_args_two_stages(args)

if args.is_training > 0:
    for ii, seed in enumerate(iteration_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # add experiment id
        setting = f'{base_setting}_{model_setting}_{args.des}_{ii}'

        # skip if already tested
        if os.path.exists(args.result_file):
            df = pd.read_csv(args.result_file)
            result_exists = (
                (df['Model ID'] == model_id) & (df['Seed'] == seed)
                ).any()
            
            if result_exists and not args.rerun:
                logger.log(
                    "Experiment result found in test_results, skipping...",
                    console_only=True)
                continue

        logger.log(f'Random seed: {seed}', level='debug')

        if args.is_training > 1:
            stage_name = f'{args.exp_id} Reconstruction Experiment Initialisation'
            try:
                exp_recon = Exp_Recon(args_recon, config, logger)
            except Exception as e:
                logger.stage_failed(e, stage_name)
                torch.cuda.empty_cache()
                continue
            stage_name = f'{args.exp_id} Reconstructor Training_{ii}'
            logger.stage_start(stage_name, setting)
            try:
                exp_recon.train(setting)
                logger.stage_end(stage_name)
            except Exception as e:
                logger.stage_failed(e, stage_name)
                torch.cuda.empty_cache()
                continue

        model_path = os.path.join(
            args.checkpoints, setting, 'reconstructor.pth')
        
        if args.is_training < 3:
            # TODO - design VQVAE predictor or remove this
            if args.model_recon == 'VQVAE':
                raise NotImplementedError
            stage_name = f'{args.exp_id} Prediction Experiment Initialisation'
            try:
                exp_pred = Exp_pred(args_pred, model_path, config, logger)
            except Exception as e:
                logger.stage_failed(e, stage_name)
                torch.cuda.empty_cache()
                continue

            stage_name = f'{args.exp_id} Latent Predictor Training_{ii}'
            logger.stage_start(stage_name, setting)
            try:
                exp_pred.train(setting)
                logger.stage_end(stage_name)
            except Exception as e:
                logger.stage_failed(e, stage_name)
                torch.cuda.empty_cache()
                continue

            if not args.train_only:
                stage_name = f'{args.exp_id} Latent Predictor Testing_{ii}'
                logger.stage_start(stage_name, setting)
                try:
                    exp_pred.test(
                        setting, model_id, args.exp_id,
                        exp_seed=seed)
                    logger.stage_end(stage_name)
                except Exception as e:
                    logger.stage_failed(e, stage_name)

        torch.cuda.empty_cache()
else:
    setting = f'{base_setting}_{model_setting}_{args.des}_{args.test_idx}'

    # skip if already tested
    result_path = './test_results/' + setting + '/' + 'pred_0.pdf'
    if os.path.exists(result_path) and not args.rerun:
        logger.log(
            "Experiment result found in test_results, skipping...",
            console_only=True)
        sys.exit()

    # path to the reconstruction model
    model_path = os.path.join(
        args.checkpoints, setting, 'reconstructor.pth')

    stage_name = f'{args.exp_id} Prediction Experiment Initialisation'
    try:
        exp_pred = Exp_pred(args_pred, model_path, config, logger)
    except Exception as e:
        logger.stage_failed(e, stage_name)
        torch.cuda.empty_cache()
        sys.exit()

    stage_name = f'{args.exp_id} Latent Predictor Testing'
    logger.stage_start(stage_name, setting)
    try:
        exp_pred.test(
            setting, model_id, args.exp_id,
            exp_seed=iteration_seeds[args.test_idx],
            load=True)
        logger.stage_end(stage_name)
    except Exception as e:
        logger.stage_failed(e, stage_name)

    torch.cuda.empty_cache()
