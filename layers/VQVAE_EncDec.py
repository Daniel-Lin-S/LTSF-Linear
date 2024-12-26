"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from utils.time_freq import timefreq_to_time, time_to_timefreq, SnakeActivation


def get_stride(downsampling: str, downsample_ratio: int=2):
    """
    Auxiliary function for converting downsampling mode
    to the stride used for VQVAE blocks

    Parameters
    ----------
    downsampling: str
        The mode of down-sampling, must be one of
        - 'time': temporal downsampling (second axis)
        - 'freq': frequency downsampling (first axis)
        - 'both': temporal and frequency down-sampling
    downsample_ratio: int, optional
        Rate of down-sampling,
        default is 2.
    """
    if downsampling == 'time':
        return (1, downsample_ratio)
    elif downsampling == 'freq':
        return (downsample_ratio, 1)
    elif downsampling == 'both':
        return (downsample_ratio, downsample_ratio)
    else:
        raise NotImplementedError(
            'downsampling strategy must be one of '
            "['time', 'freq', 'both']"
        )

class ResBlock(nn.Module):
    """
    A shape-preserving residual block for
    VQVAE with flexible kernel size.

    Uses two convolutional layers with Snake activation,
    batch normalization in bewteen and dropout at the end.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 frequency_bandwidth: int,
                 mid_channels: Optional[int]=None,
                 dropout:float=0.):
        """
        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output feature channels.
        frequency_bandwidth : int
            Kernel size in the frequency dimension. \n
            Controls the receptive field in the frequency axis.
        mid_channels : int, optional
            Number of intermediate feature channels
            in the residual block. \n
            If not given, it is set to `out_channels`.
        dropout : float, optional
            Dropout rate applied after the
            second convolutional layer. \n
            Default is 0.
        """
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        # Define kernel size and padding
        time_width = 3
        kernel_size = (frequency_bandwidth, time_width)
        freq_padding = (frequency_bandwidth - 1) // 2
        time_padding = (time_width - 1) // 2
        padding = (freq_padding, time_padding)

        layers = [
            SnakeActivation(in_channels, 2), #nn.LeakyReLU()
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2),  #nn.LeakyReLU()
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.Dropout(dropout)
        ]
        self.convs = nn.Sequential(*layers)
        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:  # adjust number of channels for skip connection
            self.proj = nn.Conv2d(
                in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.proj(x) + self.convs(x)
        return out


class VQVAEEncBlock(nn.Module):
    """
    Down-sampling block using Convolution,
    batch normalisation, snake activation and dropout. \n

    Only temporal-axis will be down-sampled (halved)
    all paddings are set to preserve shape. \n
    Bandwidth (kernel size) on temporal dimension
    is always 4; the kernel size along the frequency dimension
    can be controlled by frequency_bandwidth.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 frequency_bandwidth: int,
                 downsampling: str='time',
                 downsample_ratio: int=2,
                 dropout:float=0.
                 ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output feature channels.
        frequency_bandwidth : int
            Kernel size in the frequency dimension. \n
            Controls the receptive field in the frequency axis.
        downsampling : str, optional
            The axis for which down-sampling should occur. \n
            Must be one of 'time', 'freq', 'both'. \n
            Default is time.
        downsample_ratio : int, optional
            the stride (downsampling ratio) used. \n
            default is 2.
        dropout : float, optional
            Dropout rate applied after the activation function. \n
            Default is 0.
        """
        super().__init__()
        
        # set the kernel size and paddings
        time_width = 4
        kernel_size = (frequency_bandwidth, time_width)
        freq_padding = (frequency_bandwidth - 1) // 2
        time_padding = (time_width-1) // 2
        padding = (freq_padding, time_padding)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=get_stride(downsampling, downsample_ratio),
                      padding=padding,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #nn.LeakyReLU()
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        VQVAEEncBlock._check_input(x)
        out = self.block(x)
        return out
    
    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise TypeError(
                'x must be 4-dimensional tensor with shape '
                '(batch_size, channels, frequencies, length)'
            )
        if x.shape[3] < 2:
            raise TypeError(
                'length of temporal axis (axis 4) shorter than 2'
                ', cannot use the kernel.')


class VQVAEDecBlock(nn.Module):
    """
    Up-sampling block using transposed Convolution,
    batch normalisation, snake activation and dropout.

    Only temporal-axis will be up-sampled (doubled)
    all paddings are set to preserve shape. \n
    Bandwidth (kernel size) on temporal dimension
    is always 4; the kernel size along the frequency dimension
    can be controlled by frequency_bandwidth.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 frequency_bandwidth: int,
                 downsampling: str='time',
                 downsample_ratio: int=2,
                 dropout:float=0.
                 ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output feature channels.
        frequency_bandwidth : int
            Kernel size in the frequency dimension. \n
            Controls the receptive field in the frequency axis. \n
            Set it to 1 for independent frequency bands.
        downsampling : str, optional
            The axis for which down-sampling should occur. \n
            Must be one of 'time', 'freq', 'both'. \n
            Default is time.
        downsample_ratio : int, optional
            the stride (downsampling ratio) used. \n
            default is 2.
        dropout : float, optional
            Dropout rate applied after the activation function. \n
            Default is 0.
        """
        super().__init__()
        
        # set the kernel size and paddings
        time_width = 4
        kernel_size = (frequency_bandwidth, time_width)
        freq_padding = (frequency_bandwidth - 1) // 2
        time_padding = (time_width-1) // 2
        padding = (freq_padding, time_padding)

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=get_stride(downsampling, downsample_ratio),
                padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #nn.LeakyReLU()
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.

    Components
    ----------
    Down-sampling blocks (VQVAEEncBlock):
    doubled channels, halved temporal length. \n
    Residual block (ResBlock):
    residual connection, number of channels
    preserved except for the last layer
    to match required hidden dimension size (hid_dim).
    
    x -> STFT -> VQVAEEncBlock -> [VQVAEEncBlock ->
    [ResBlock * n_resnet_blocks]] * m
    -> ResBlock -> z \n
    where m = log2(downsample_rate) - 1.
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 pad_func: callable,
                 n_fft:int,
                 frequency_bandwidth:int,
                 downsampling: str='time',
                 dropout: float=0.3,
                 stft_split_type: str='real_imag'
                 ):
        """
        Parameters
        -----------
        init_dim : int
            Number of output channels of the
            first convolution block.
        hid_dim : int
            Hidden dimension size foreach time-frequency point.
        num_channels : int
            Channel size of the input.
        downsampling : str, optional
            The axis for which down-sampling should occur. \n
            Must be one of 'time', 'freq', 'both'. \n
            Default is time.
        downsample_rate : int
            Spatial downsampling factor for the temporal axis.
            Should be a power of 2; e.g., 2, 4, 8, 16, ...
        n_resnet_blocks : int
            Number of ResNet blocks between each downsampling block.
        pad_func : callable
            a function that takes a tensor and
            modifies it to mask some frequency bands
            or perform other operations. \n
            Set to None to skip this.
        n_fft : int
            numebr of FFT points to use.
        frequency_bandwidth : int
            Kernel size in the frequency dimension. \n
            Controls the receptive field in the frequency axis.
        dropout : float, optional
            global dropout rate.
            Default is 0.3
        stft_split_type: str, optional
            The method used to store complex frequency components
            of STFT. See details in `utils.time_to_timefreq`.
        """
        super().__init__()
        self.pad_func = pad_func
        self.n_fft = n_fft

        if stft_split_type not in ['real_imag', 'magnitude_phase', 'none']:
            raise ValueError(
                'stft_split_type must be one of real_image, '
                'magnitude_phase and none')
        self.stft_split_type = stft_split_type

        ### hierarchical en-coding ###
        d = init_dim
        enc_layers = [
            VQVAEEncBlock(num_channels, d, frequency_bandwidth, downsampling),]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(
                VQVAEEncBlock(d//2, d, frequency_bandwidth, downsampling))
            for _ in range(n_resnet_blocks):
                enc_layers.append(
                    ResBlock(d, d, frequency_bandwidth, dropout=dropout))
            d *= 2
        enc_layers.append(
            ResBlock(d//2, hid_dim, frequency_bandwidth, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        # to store height and width of the output 
        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.tensor(0))
        self.register_buffer('H_prime', torch.tensor(0))
        self.register_buffer('W_prime', torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform STFT and then process time-frequency
        image via PixelCNN. (as defined in init)
        :param x: tensor of shape (batch_size channels length)
        """
        in_channels = x.shape[1]
        x = time_to_timefreq(
            x, self.n_fft, in_channels,
            split_type=self.stft_split_type)  # (b c h w)
        if self.pad_func:  # used to cover some frequency bands
            x = self.pad_func(x, copy=True)   # (b c h w)

        out = self.encoder(x)  # (b c h w)

        # store height and width of the output 
        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True

        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.

    
    Components
    ----------
    Up-sampling blocks (VQVAEDecBlock):
    halved channels, doubled temporal length.\n
    Residual block (ResBlock):
    residual connection, number of channels
    preserved except for the last layer
    to match required hidden dimension size (hid_dim).
    
    z -> ResBlock -> [[ResBlock * n_resnet_blocks]
    -> VQVAEDecBlock] * m
    -> ConvT * 2 -> iSTFT -> x \n
    where m = log2(downsample_rate) - 1.

    Notes
    -----
    The following parameters must be the same
    as VQVAEEncoder:
    init_dim, hid_dim, n_fft. \n
    The following are suggested to be consistent
    with VQVAEEncoder:
    number_channels, downsample_rate,
    frequency_bandwidth. \n
    The following must be provided for reconstruction:
    x_channels, input_length
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 input_length:int,
                 pad_func: callable,
                 n_fft:int,
                 x_channels:int,
                 frequency_bandwidth:int,
                 downsampling: str='time',
                 dropout:float=0.3,
                 stft_split_type: str='real_imag'):
        """
        Parameters
        -----------
        init_dim : int
            Number of output channels of the
            first convolution block in VQVAEEncoder
        hid_dim : int
            Hidden dimension size for each time-frequency point.
            (same as VQVAEEncoder)
        num_channels : int
            Channel size of the input.
        downsample_rate : int
            Spatial downsampling factor for the temporal axis.
            Should be a power of 2; e.g., 2, 4, 8, 16, ...
        n_resnet_blocks : int
            Number of ResNet blocks between each downsampling block.
        input_length : int
            length of the original time-series
        pad_func : callable
            a function that takes a tensor and
            modifies it to mask some frequency bands
            or perform other operations. \n
            The output shape must be the same!
            Set to None to skip this.
        n_fft : int
            numebr of FFT points used in VQVAEEncoder
        x_channels : int
            number of channels of the original
            time-series.
        frequency_bandwidth : int
            Kernel size in the frequency dimension. \n
            Controls the receptive field in the frequency axis.
        dropout : float
            global dropout rate
        stft_split_type: str, optional
            The method used to store complex frequency components
            of STFT. See details in `utils.time_to_timefreq`.
        """
        super().__init__()
        self.pad_func = pad_func
        self.n_fft = n_fft
        self.x_channels = x_channels

        if stft_split_type not in ['real_imag', 'magnitude_phase', 'none']:
            raise ValueError(
                'stft_split_type must be one of real_image, '
                'magnitude_phase and none')
        self.stft_split_type = stft_split_type

        ### hierarchical de-coding ###
        # note enc_out_dim == dec_in_dim
        d = int(init_dim * 2**(int(round(np.log2(downsample_rate))) - 1))  
        if round(np.log2(downsample_rate)) == 0:
            d = int(init_dim * 2**(int(round(np.log2(downsample_rate)))))
        dec_layers = [ResBlock(hid_dim, d, frequency_bandwidth, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_bandwidth, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2*d, d, frequency_bandwidth, downsampling))

        ### restore resolution ###
        # set the kernel size and paddings
        time_width = 4
        kernel_size = (frequency_bandwidth, time_width)
        freq_padding = (frequency_bandwidth - 1) // 2
        time_padding = (time_width-1) // 2
        padding = (freq_padding, time_padding)

        recovery_stride = get_stride(downsampling)

        dec_layers.append(
            nn.ConvTranspose2d(d, num_channels,
                            kernel_size=kernel_size,
                            stride=recovery_stride, padding=padding))
        if downsampling == 'time':
            dec_layers.append(
                nn.ConvTranspose2d(num_channels, num_channels,
                                kernel_size=kernel_size,
                                stride=recovery_stride, padding=padding))
        self.decoder = nn.Sequential(*dec_layers)

        self.interp = nn.Upsample(input_length, mode='linear')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(x)  # (b c h w)
        if self.pad_func:
            out = self.pad_func(out)  # (b c h w)
        out = timefreq_to_time(
            out, self.n_fft, self.x_channels,
            split_type=self.stft_split_type)  # (b c l)

        out = self.interp(out)  # (b c l)
        return out
