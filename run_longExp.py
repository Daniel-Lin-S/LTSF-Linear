import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import os
import sys
from utils.logger import DualLogger
from utils.configs import get_base_settings, get_pred_model_settings


### Set Global seed for reproducibility ###
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


### Collect Arguments ###
parser = argparse.ArgumentParser(
    description='Time Series Forecasting Experiment settings')

# basic config
parser.add_argument('--is_training', type=int, required=True, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False,
                    help='perform training on full input dataset without validation and testing')
parser.add_argument('--test_idx', type=int, default=0,
                    help='index of the experiment repetition used for testing'
                    ' if is_training = 0')
parser.add_argument('--model_id', type=str, required=True, help='model id')
parser.add_argument('--model', type=str, required=True,
                    help='model name, options: [Autoformer, Informer, Transformer, '
                    'DLinear, Linear, NLinear, FDLinear]')
parser.add_argument('--itr', type=int, default=2, help='number of experiment repetitions')
parser.add_argument('--des', type=str, default='',
                    help='experiment description added at the end of folder name')
parser.add_argument('--log_file', type=str, default='logs/test.log',
                    help='file path of log file for the training process')
parser.add_argument('--do_predict', action='store_true',
                    help='whether to predict unseen future data')
parser.add_argument('--rerun', action='store_true', default=False,
                    help='whether to cover existing test results')

# data loader
parser.add_argument('--data', type=str, required=True, help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                    'M:multivariate predict multivariate, '
                    'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, '
                    't:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                    'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') # 96=24*4
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--hop_length', type=int, default=1,
                    help='hop length for the sliding window of getting dataset '
                    'of the prediction model')

# DLinear
parser.add_argument('--individual', action='store_true', default=False,
                    help='For linear models, a linear layer for each variate(channel) individually')

# FDLinear and STFTLinear
parser.add_argument('--stft_hop_length', type=int, default=4,
                    help='FDLinear: hop length of sliding window for STFT')
parser.add_argument('--nfft', type=int, default=8, help='FDLinear: number of FFT points')

# Formers 
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default '
                    '1: value embedding + temporal embedding + positional embedding '
                    '2: value embedding + temporal embedding '
                    '3: value embedding + positional embedding 4: value embedding')
# Linear models with --individual should use enc_in as the number of channels
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
                    help='whether to use distilling in encoder, '
                    'using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention',
                    action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

### Prepare Logger ###
logger = DualLogger(args.log_file)

logger.start_experiment(args.model_id, args.is_training)
logger.log(f'Experiment settings: \n {args}', level='debug')

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

### Set seeds for iterations ###
iteration_seeds = [random.randint(0, 2**32 - 1) for _ in range(args.itr)]
logger.log(f'Random seeds for experiments: {iteration_seeds}', level='debug')

### Record basic and model settings ###
base_setting = get_base_settings(args)
model_setting = get_pred_model_settings(args)

### store model hyperparameters ###
if args.is_training:
    for ii, seed in enumerate(iteration_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        setting = f'{base_setting}_{model_setting}_{args.des}_{ii}'

        # skip if already tested
        result_path = './test_results/' + setting + '/' + 'pred_0.pdf'
        if os.path.exists(result_path) and not args.rerun:
            logger.log(
                "Experiment result found in test_results, skipping...",
                console_only=True)
            continue

        # Training Stage
        exp = Exp_Main(args, logger)
        stage_name = f'{args.model_id} Predictor Training_{ii}'
        logger.start_stage(stage_name, setting)
        exp.train(setting)
        logger.finish_stage(stage_name)

        # Testing Stage
        if not args.train_only:
            stage_name = f'{args.model_id} Testing_{ii}'
            logger.start_stage(stage_name, setting)
            exp.test(setting)
            logger.finish_stage(stage_name)

        # Prediction Stage
        if args.do_predict:
            stage_name = f'{args.model_id} Prediction_{ii}'
            logger.start_stage(stage_name, setting)
            exp.predict(setting, True)
            logger.finish_stage(stage_name)

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

    exp = Exp_Main(args, logger)

    if args.do_predict:  # Prediction Stage
        stage_name = f'{args.model_id} Prediction'
        logger.start_stage(stage_name, setting)
        exp.predict(base_setting, True)
        logger.finish_stage(stage_name)
    else:  # Test Stage
        stage_name = f'{args.model_id} Testing'
        logger.start_stage(stage_name, setting)
        exp.test(base_setting, test=1)
        logger.finish_stage(stage_name)
    torch.cuda.empty_cache()
