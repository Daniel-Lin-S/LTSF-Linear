import argparse
from copy import deepcopy


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

def get_base_settings(args: argparse.Namespace) -> str:
    """
    Extract basic training settings into a string.
    """
    base_setting = '{}_{}_{}_ft{}_eb{}_sl{}_ll{}_pl{}'.format(
        args.exp_id,
        args.model,
        args.data,
        args.features,
        args.embed,
        args.seq_len,
        args.label_len,
        args.pred_len,
    )

    return base_setting

def get_pred_model_settings(args: argparse.Namespace) -> str:
    """
    Generate unique model id for prediction model based on
    the hyperparameters. \n
    
    Used to name the folders storing checkpoints and test
    results.
    """
    if 'Linear' in args.model:
        base_model_setting = 'ind{}'.format(
            args.individual
        )
        if args.model == 'FDLinear':
            additional_setting = 'nfft{}_hoplen{}'.format(
                args.nfft,
                args.stft_hop_length
            )
        elif args.model == 'STFTLinear':
            additional_setting = 'freqind{}_nfft{}_hoplen{}'.format(
                args.independent_freqs,
                args.nfft,
                args.stft_hop_length
            )
        else:
            additional_setting = None
    elif 'former' in args.model or args.model == 'PatchTST':
        base_model_setting = 'dim{}_nheads{}_enclayers{}_df{}_dropout{}'.format(
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.dropout
        )
        if args.model == 'Transformer':
            additional_setting = 'declayers{}_embedtype{}_act{}_attnfactor{}'.format(
                args.d_layers,
                args.embed_type,
                args.activation,
                args.factor,
            )
        elif args.model == 'Informer':
            additional_setting = 'declayers{}_embedtype{}_act{}_attnfactor{}_distil{}'.format(
                args.d_layers,
                args.embed_type,
                args.activation,
                args.factor,
                args.distil
            )
        elif args.model == 'Autoformer':
            additional_setting = 'declayers{}_embedtype{}_act{}_attnfactor{}_mavg{}'.format(
                args.d_layers,
                args.embed_type,
                args.activation,
                args.factor,
                args.moving_avg
            )
        elif args.model == 'FEDformer':
            additional_setting = 'declayers{}_embedtype{}_act{}_version{}_modes{}_select{}'.format(
                args.d_layers,
                args.embed_type,
                args.activation,
                args.version,
                args.modes,
                args.mode_select
            )
            if args.version == 'Wavelets':
                additional_setting += '_wvt{}_wvtskips{}_crossact{}'.format(
                    args.base,   # the wavelet used
                    args.L,      # the number of wavelet scales to skips
                    args.cross_activation
                )
        elif args.model == 'PatchTST':
            additional_setting = 'ind{}_fcdropout{}_headdropout{}_patchlen{}_revin{}_affine{}_sublast{}_decomp{}_mvgkernel{}'.format(
                args.individual,
                args.fc_dropout,
                args.head_dropout,
                args.patch_len,
                args.revin,
                args.affine,
                args.subtract_last,
                args.decomposition,
                args.kernel_size
            )
        else:
            additional_setting = None
        
    if additional_setting:
        return f'{base_model_setting}_{additional_setting}'
    else:
        return base_model_setting

def get_recon_model_settings(args, config: dict) -> str:
    if args.model_recon == 'VAE':
        if config['encoder']['downsampling'] == 'time':
            width_l = config['encoder']['downsampled_width']['lf']
            width_h = config['encoder']['downsampled_width']['hf']
            latent_width = f'lf{width_l}_hf{width_h}'
        elif config['encoder']['downsampling'] == 'freq':
            latent_width = config['encoder']['downsampled_width']['freq']

        model_setting = 'hd{}_fb{}_lw{}_ld{}_lt[{}]_beta{}->{}_nfft{}_split[{}]_sep[{}]'.format(
            config['encoder']['hid_dim'],
            config['encoder']['frequency_bandwidth'],
            latent_width,
            config['vae']['latent_dim'],
            config['vae']['latent_type'],
            config['vae']['beta_init'],
            config['vae']['beta'],
            config['n_fft'],
            config['stft_split_type'],
            config['lfhf_separation']
        )
    elif args.model_recon == 'AE':
        if config['encoder']['downsampling'] == 'time':
            width_l = config['encoder']['downsampled_width']['lf']
            width_h = config['encoder']['downsampled_width']['hf']
            latent_width = f'lf{width_l}_hf{width_h}'
        elif config['encoder']['downsampling'] == 'freq':
            latent_width = config['encoder']['downsampled_width']['freq']
        
        model_setting = 'hd{}_fb{}_lw{}_nfft{}_split[{}]_sep[{}]'.format(
            config['encoder']['hid_dim'],
            config['encoder']['frequency_bandwidth'],
            latent_width,
            config['n_fft'],
            config['stft_split_type'],
            config['lfhf_separation']
        )

    elif args.model_recon == 'VQVAE':
        if config['encoder']['downsampling'] == 'time':
            width_l = config['encoder']['downsampled_width']['lf']
            width_h = config['encoder']['downsampled_width']['hf']
            latent_width = f'lf{width_l}_hf{width_h}'
        elif config['encoder']['downsampling'] == 'freq':
            latent_width = config['encoder']['downsampled_width']['freq']
        
        model_setting = 'hd{}_fb{}_lw{}_nfft{}_split[{}]_sep[{}]_cblf{}_cbhf{}_cbarg{}'.format(
            config['encoder']['hid_dim'],
            config['encoder']['frequency_bandwidth'],
            latent_width,
            config['n_fft'],
            config['stft_split_type'],
            config['lfhf_separation'],
            config['VQ-VAE']['codebook_sizes']['lf'],
            config['VQ-VAE']['codebook_sizes']['hf'],
            config['VQ-VAE']['codebook']
        )
    else:
        raise NotImplementedError

    return model_setting
