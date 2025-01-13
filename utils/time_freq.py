import numpy as np
import torch
import torch.nn.functional as F
import torch.jit as jit
import torch.nn as nn
from typing import Tuple, Optional, Union
from einops import rearrange, repeat
import matplotlib.pyplot as plt


def compute_downsample_rate(input_length: int,
                            n_fft: int,
                            downsampled_width: int,
                            downsampling: str='time',
                            return_width: bool=False) -> int:
    """
    Compute the downsampling rate for
    the PixelCNN architecture in VQVAE.

    :param input_length: length of input sequence
    :param n_fft: number of FFT points used in STFT
    :param downsampled_width: the temporal width of encoded vector.
      Actual width might be slightly longer due to rounding
    :param downsampling: strategy of downsampling
      one of ['time', 'freq']
    :param return_width: if true, return
      the estimated actual width of encoded vector.

    :return int: corresponding downsampling rate.
    :return tuple: actual down-sampled height and width,
      if return_size=True

    Notes
    -----
    - This fuction assumes STFT uses default hop length
    - and downsampling ratio is 2.
    """
    # default hop length assumed
    stft_length = np.ceil(
        (input_length - n_fft) / np.floor(n_fft / 4)) + 5
    
    freqs = round(n_fft / 2) + 1  # number of frequency bins

    if downsampling == 'time':
        init_width = stft_length
    elif downsampling == 'freq':
        init_width = freqs
    else:
        raise NotImplementedError(
            'downsampling strategy other than'
            "'time' and 'freq' is not implemented yet"
        )

    downsample_rate = round(
        init_width / downsampled_width) if (
            stft_length >= downsampled_width) else 1

    downsample_rounds = max(1, int(round(np.log2(downsample_rate))))
    actual_downsampled_width = round(init_width / (2**downsample_rounds))

    # for the shortest possible length, convolution is handled differently.
    if actual_downsampled_width == 4:
        if downsampling == 'time':
            actual_downsampled_width -= 1
        if downsampling == 'freq':
            actual_downsampled_width += 1

    if return_width:
        return downsample_rate, actual_downsampled_width
    else:
        return downsample_rate
    

def time_to_timefreq(x: torch.Tensor, n_fft: int,
                     C: int, norm:bool=True,
                     split_type: str='real_imag',
                     stft_kwargs: Optional[dict]=None):
    """
    Arguments
    ----------
    x : torch.Tensor
        of shape (B, C, L). \n
        B - batch size,
        C - number of channels,
        L - length
    n_fft : int
        number of FFT points to use
    C : int
        number of channels of x
    norm : bool, optional
        if True, perform normalised STFT
    split_type : str, optional
        Defines how the complex-valued frequency
        components will be split:
        - 'real_imag' : splits into real and
          imaginary parts (default).
        - 'magnitude_phase' : splits into
          magnitude and phase components.
        - 'none' : does not split the complex values,
          i.e., keeps them as complex.
    stft_kwargs : dict
        additional arguments passed to torch.stft,
        e.g. hop_length

    Return
    ------
    Tensor 
        of shape (B, 2C, H, W). \n
        H, W: frequency and time dimensions
        2C used to store complex numbers. \n
        When split_type = 'none', return
        shape is (B, C, H, W) but complex-
        valued.
    
    Notes
    -----
    Since the input is always real,
    there are redundant frequencies
    and only floor(n_fft/2)+1 frequency
    channels will produced. \n
    Real and imagery part of frequency components
    are dealt separately,
    i.e. in two separate channels.
    """
    if x.ndim != 3:
        raise TypeError(
            'input x must have three dimensions: '
            '(batch_size, channels, length), '
            f'got {x.shape}')
    if x.shape[2] < n_fft:
        raise ValueError(
            f'The length of signal, {x.shape[2]}, '
            f'must be longer than n_fft, {n_fft}')

    x = rearrange(x, 'b c l -> (b c) l')
    window = torch.hann_window(window_length=n_fft, device=x.device)

    if stft_kwargs:
        x = torch.stft(x, n_fft, normalized=norm,
                       return_complex=True,
                       window=window, **stft_kwargs)
    else:
        x = torch.stft(x, n_fft, normalized=norm,
                       return_complex=True,
                       window=window)

    if split_type == 'real_imag':
        # Split into real and imaginary parts
        x = torch.view_as_real(x)
        x = rearrange(
            x, '(b c) n t z -> b (c z) n t', c=C)  # z=2 (real, imag)
    elif split_type == 'magnitude_phase':
        # Split into magnitude and phase (argument)
        magnitude = torch.abs(x)
        argument = torch.angle(x)

        x = torch.stack([magnitude, argument], dim=-3)
    elif split_type == 'none':
        # return the complex-valued tensor as-is
        x = rearrange(x, '(b c) n t -> b c n t', c=C)
    else:
        raise ValueError(
            'split_type must be one of real_imag, '
            'magnitude_phase, none')

    return x


def timefreq_to_time(x: torch.Tensor, n_fft: int, C: int,
                     split_type: str = 'real_imag', 
                     norm:bool=True,
                     stft_kwargs: Optional[dict]=None) -> torch.Tensor:
    """
    Convert time-frequency domain tensor (STFT representation) back to the time domain.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in the time-frequency domain.
    n_fft : int
        The FFT size used during the STFT.
    C : int
        The number of input channels.
    split_type : str, optional
        how to interpret the frequency component representation:
        - 'real_imag' : channels have real and imaginary parts (default).
        - 'magnitude_phase' : channels have magnitude and phase components.
        - 'none' : complex-valued.
    norm : bool, optional
        Whether to normalize the ISTFT (default: True).
    stft_kwargs : dict
        additional arguments passed to torch.istft,
        e.g. hop_length
    
    Return
    ------
    torch.Tensor
        the restored time domain representation of x.
    """
    # Re-arrange input tensor according to the split type
    if split_type == 'real_imag':
        # Real and Imaginary split
        x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
        x = torch.view_as_complex(x)
    elif split_type == 'magnitude_phase':
        # Magnitude and Phase split
        x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
        magnitude, phase = x[..., 0], x[..., 1]
        # Rebuild the complex representation
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        x = torch.stack([real, imag], dim=-1)
        x = torch.view_as_complex(x)
    elif split_type == 'none':
        x = x.contiguous()
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")
    
    window = torch.hann_window(window_length=n_fft, device=x.device)

    if stft_kwargs:
        x = torch.istft(x, n_fft, normalized=norm,
                       window=window, **stft_kwargs)
    else:
        x = torch.istft(x, n_fft, normalized=norm,
                       window=window)
    x = rearrange(x, '(b c) l -> b c l', c=C)

    return x
    

def zero_pad_high_freq(xf: torch.Tensor,
                       copy: bool=False) -> torch.Tensor:
    """
    Fill all frequencies other than mean energy (component 0)
    with zeroes or copies of the mean energy.

    Parameters
    ----------
    xf : torch.Tensor
        of shape (B, C, H, W);
        H: frequency-axis, W: temporal-axis
    copy : bool, optional
        if true, copy the mean energy (component 0)
        to other frequency components.
        Default is False
    
    Return
    ------
    torch.Tensor
        the padded time-frequency representation xf
        of the same shape as xf.
    """
    if not copy:
        xf_l = torch.zeros(xf.shape).to(xf.device)
        xf_l[:, :, 0, :] = xf[:, :, 0, :]  # (b c h w)
    else:
        # model input: copy the LF component and 
        # paste it to the rest of the frequency bands
        xf_l = xf[:, :, [0], :]  # (b c 1 w)
        xf_l = repeat(xf_l, 'b c 1 w -> b c h w', h=xf.shape[2]).float()  # (b c h w)
    return xf_l

def zero_pad_low_freq(xf: torch.Tensor,
                       copy: bool=False) -> torch.Tensor:
    """
    Fill the mean energy (component 0)
    with zeroes or a copy of component 1.

    Parameters
    ----------
    xf : torch.Tensor
        of shape (B, C, H, W);
        H: frequency-axis, W: temporal-axis
    copy : bool, optional
        if true, copy the mean energy (component 0)
        to other frequency components.
        Default is False
    
    Return
    ------
    torch.Tensor
        the padded time-frequency representation xf,
        of the same shape as xf.
    """
    if not copy:
        xf_h = torch.zeros(xf.shape).to(xf.device)
        xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    else:
        # model input: copy the first HF component,
        # and paste it to the LF band
        xf_h = xf[:, :, 1:, :]  # (b c h-1 w)
        xf_h = torch.cat((xf_h[:,:,[0],:], xf_h), dim=2).float()
    return xf_h


def stft_lfhf(x: torch.Tensor,
              n_fft: int,
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply STFT to a batch of time series and
    return the LF (mean energy) and HF components (oscillating patterns)
    respectively.

    Parameters
    ----------
    x : torch.Tensor
        of shape (batch_size, channels, length),
        the time series samples
    n_fft : int
        Size of FFT used for STFT
    kwargs : Any
        additional arguments passed to
        time_to_timefreq and timefreq_to_time.
        Note: number of channels will be calculated
        automatically.

    Returns
    -------
    x_l, x_h: torch.Tensor
        same shape as input x, the rebuilt
        time serieses with
        low-frequency and high-frequency.
    """
    in_channels = x.shape[1]
    input_length = x.shape[2]
    xf = time_to_timefreq(x, n_fft, in_channels, **kwargs)  # (b c h w)
    u_l = zero_pad_high_freq(xf)  # (b c h w)
    x_l = F.interpolate(
        timefreq_to_time(u_l, n_fft, in_channels, **kwargs),
        input_length, mode='linear')  # (b c l)
    u_h = zero_pad_low_freq(xf)  # (b c h w)
    x_h = F.interpolate(
        timefreq_to_time(u_h, n_fft, in_channels, **kwargs),
        input_length, mode='linear')  # (b c l)

    return x_l, x_h

def plot_lfhf_reconstruction(x_l, xhat_l, x_h, xhat_h,
                             step: Optional[int]=None,
                             file_path: Optional[str]=None) -> None:
    """
    Plot reconstruction results for low-frequency (LF)
    and high-frequency (HF) components.

    Parameters:
    ----------
    x_l : torch.Tensor
        Ground truth low-frequency component, shape (B, C, L).
    xhat_l : torch.Tensor
        Reconstructed low-frequency component, shape (B, C, L).
    x_h : torch.Tensor
        Ground truth high-frequency component, shape (B, C, L).
    xhat_h : torch.Tensor
        Reconstructed high-frequency component, shape (B, C, L).
    step : int, optional
        Current training step or epoch number for labeling.
    file_path : str, optional
        If given, the figure will be saved instead
        of shown directly.

    Returns:
    -------
    None
    """
    # Randomly select a batch and channel
    b = np.random.randint(0, x_h.shape[0])
    c = np.random.randint(0, x_h.shape[1])

    # basic set-ups
    alpha = 0.7  # transparency
    n_rows = 3  # LF, HF, combined
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2 * n_rows))
    title = f"channel idx:{c} \n (blue:Ground Truth, orange:reconstructed)"
    if step is not None:
        title = f"step-{step} | {title}"
    plt.suptitle(title)

    # Plot low-frequency component
    u_l = x_l[b, c].cpu()
    u_l_recon = xhat_l[b, c].detach().cpu()
    axes[0].plot(u_l, alpha=alpha)
    axes[0].plot(
        u_l_recon, alpha=alpha)
    axes[0].set_title(r'$x_l$ (LF)')
    y_min_l, y_max_l = min(u_l.min(), u_l_recon.min()), \
                    max(u_l.max(), u_l_recon.max())
    axes[0].set_ylim(y_min_l - 0.1, y_max_l + 0.1)

    # Plot high-frequency component
    u_h = x_h[b, c].cpu()
    u_h_recon = xhat_h[b, c].detach().cpu()
    axes[1].plot(u_h, alpha=alpha)
    axes[1].plot(
        u_h_recon, alpha=alpha)
    axes[1].set_title(r'$x_h$ (HF)')
    y_min_h, y_max_h = min(u_h.min(), u_h_recon.min()), \
                    max(u_h.max(), u_h_recon.max())
    axes[1].set_ylim(y_min_h - 0.1, y_max_h + 0.1)

    # Plot combined components
    u = x_l[b, c].cpu() + x_h[b, c].cpu()
    u_recon = xhat_l[b, c].detach().cpu() + xhat_h[b, c].detach().cpu()
    axes[2].plot(
        u, alpha=alpha)
    axes[2].plot(u_recon,
                 alpha=alpha)
    axes[2].set_title(r'$x$ (LF+HF)')
    y_min, y_max = min(u.min(), u_recon.min()), \
                    max(u.max(), u_recon.max())
    axes[2].set_ylim(y_min - 0.1, y_max + 0.1)

    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


class SnakeActivation(jit.ScriptModule):
    """
    Snake activation: a non-linear activation
    that adds sine-squared to the input. \n
    The activation function is defined as:
    
        f(x) = x + (1 / a) * sin(a * x)^2

    This implementation allows multiple values of `a`
    for different channels. (num_features)

    Attributes
    ----------
    a : torch.Tensor
        The learnable or fixed parameter controlling the
        frequency and amplitude of the sinusoidal component. \n
        Its shape depends on the input dimension (`dim`).
    """
    def __init__(self, num_features:int, dim:int, a_base=0.2, learnable=True, a_max=0.5):
        """
        Parameters
        ----------
        num_features : int
            The number of features (channels) in the input data.
        dim : int
            The dimensionality of the input data. \n
            Must be 1 (e.g., time series) or 2 (e.g., images).
        a_base : float, optional
            The fixed value for the parameter `a` when `learnable=False`
            or as the lower bound for random initialization when
            `learnable=True`. \n
            Default is 0.2.
        learnable : bool, optional
            Whether the parameter `a` should be learnable. \n
            Default is True.
        a_max : float, optional
            The upper bound for random initialization of `a`
            when `learnable=True`. \n
            Default is 0.5.
        """
        super().__init__()
        assert dim in [1, 2], '`dim` supports 1D and 2D inputs.'

        # prepare the parameter a
        if learnable:
            if dim == 1:  # input has shape (b num_features l); like time series
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1))  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2: # input has shape (b num_features h w); like image
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1, 1))  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else: # fixed value
            self.register_buffer('a', torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            must be of shape (batch_size, num_features, length)
            or (batch_size, num_features, h, w)
            where h w are spatial dimensions
            when input is an image.
        """
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2

def quantize(z: torch.Tensor, vq_model,
             transpose_channel_length_axes: bool=False,
             svq_temp:Union[float,None]=None):
    """
    quantise embedding z according to whether
    it is temporal or temporal-frequency representation.

    Arguments
    ---------
    z : torch.Tensor
        of shape (b c h w) or (b c l). \n
        l is (the length of) temporal axis,
        h w are frequency and temporal axes,
        b is batch size, c is number of channels
    vq_model : VectorQuantize
        the Vector Quantisation model to be used. \n
        Note: codeword is always assigned to
        the last axis of input
    transpose_channel_length_axes : bool, optional
        only for time representation. \n
        If true, each channel will be assigned a codeword; \n
        otherwise, each time stamp is assigned a codeword.
    sqv_temp: float, optional
        Draw the codeword weighted by Euclidean distance,
        higher temperature gives more uniform distribution. \n
        If not given, deterinistically chooses the closest codeword.

    Returns
    -------
    z_q : torch.Tensor
        The quantized tensor where feature vectors are
        replaced with their corresponding codewords. \n
        Output shape: (b d h w)
        or (b d l) or (b c d),
        where d is codebook dimension.
    indices : torch.Tensor
        Indices of the codewords selected for each feature vector.
    vq_loss : dict
        A dictionary containing the following keys:
        - `commit_loss`: The commitment loss, encouraging the input to
        stay close to the selected codewords.
        - `orthogonal_reg_loss`: The orthogonal regularization loss,
        ensuring codebook embeddings remain diverse.
        - `loss`: The weighted sum of `commit_loss` and `orthogonal_reg_loss`,
        used as the total quantization loss.
    perplexity : float
        A measure of codebook usage diversity.

    Raises
    ------
    ValueError
        If the input tensor's dimensionality is not 3D or 4D
    """
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        # quantise each spatial location on time-frequency image
        h, w = z.shape[2:]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, 'b c l -> b (l) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, 'b (l) c -> b c l')
    else:
        raise ValueError(
            'z must have either 3 or 4 dimensions'
        )
    return z_q, indices, vq_loss, perplexity
