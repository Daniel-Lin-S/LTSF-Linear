import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from einops import rearrange, repeat
import matplotlib.pyplot as plt

from functools import partial

from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt


class STFT:
    """
    For performing short-time Fourier transform (STFT)
    and its inverse transformation
    on a batch of time series.

    Attributes
    ----------
    n_freqs : int
        number of frequency components
    in_channels : int
        Number of input channels
    window : torch.Tensor
        The Hann window used for STFT.
    """
    def __init__(
            self, n_fft: int,
            hop_length: int,
            split_type: str='real_imag',
            normalize: bool=True
        ):
        """
        Parameters
        ----------
        n_fft : int
            Number of FFT points to use.
        hop_length : int
            Hop length for the STFT.
        split_type : str
            The method used to split
            complex numbers in STFT.
            One of 'real_imag', 
            'magnitude_phase', and 'none'.
        normalize : bool
            If True, perform normalised STFT.
        """
        super(STFT, self).__init__()
        if hop_length >= n_fft:
            raise AssertionError(
                'hop_length cannot be longer than n_fft')
        
        if split_type not in ['real_imag', 'magnitude_phase', 'none']:
            raise ValueError(
                'split_type must be one of real_imag, '
                'magnitude_phase, none')

        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.split_type = split_type
        self.normalize = normalize

        self.in_channels = None
        self.window = None

    def transform(
            self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform STFT on the input time series.

        Parameters
        ----------
        x : torch.Tensor
            of shape (batch_size, length, channels),
            the input time series.

        Returns
        -------
        torch.Tensor
            the STFT representation of x.
            Of shape (batch_size, channels, n_freqs, time)
            if split_type is 'none'. Otherwise, has shape
            (batch_size, 2*channels, n_freqs, time).
        """
        self.in_channels = x.shape[2]
        self.window = torch.hann_window(
            window_length=self.n_fft, device=x.device)

        x = rearrange(x, 'b l c -> (b c) l')

        xf = torch.stft(
            x, self.n_fft, self.hop_length,
            window=self.window, normalized=self.normalize,
            return_complex=True)
        
        if self.split_type == 'real_imag':
            # Split into real and imaginary parts
            xf = torch.view_as_real(xf)
            xf = rearrange(
                xf, '(b c) n t z -> b (c z) n t',  # z=2 (real, imag)
                c=self.in_channels)  
        elif self.split_type == 'magnitude_phase':
            # Split into magnitude and phase (argument)
            magnitude = torch.abs(xf)
            argument = torch.angle(xf)

            xf = torch.stack([magnitude, argument], dim=-3)
        elif self.split_type == 'none':
            # return the complex-valued tensor as-is
            xf = rearrange(xf, '(b c) n t -> b c n t', c=self.in_channels)

        return xf
        
    def inverse_transform(
            self, xf: torch.Tensor) -> torch.Tensor:
        """
        Perform inverse STFT on the input frequency-domain tensor.

        Parameters
        ----------
        xf : torch.Tensor
            of shape (batch_size, channels, n_freqs, time),
            the frequency-domain representation of the input.

        Returns
        -------
        torch.Tensor
            the restored time domain representation of x.
            Of shape (batch_size, length, channels).
        """
        if self.in_channels is None or self.window is None:
            raise RuntimeError(
                'Please call transform method first. '
            )
        if xf.shape[1] != 2 * self.in_channels:
            raise ValueError(
                'The second axis of xf does not match '
                'input channels when transform is called, '
                f'got {xf.shape[1]}, expecting {self.in_channels * 2}')
        
        if self.split_type == 'real_imag':
            # Real and Imaginary split
            xf = rearrange(xf, 'b (c z) n t -> (b c) n t z',
                           c=self.in_channels).contiguous()
            xf = torch.view_as_complex(xf)
        elif self.split_type == 'magnitude_phase':
            xf = rearrange(xf, 'b (c z) n t -> (b c) n t z',
                           c=self.in_channels).contiguous()
            magnitude, phase = x[..., 0], x[..., 1]
            # Rebuild the complex representation
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            xf = torch.stack([real, imag], dim=-1)
            xf = torch.view_as_complex(xf)
        elif self.split_type == 'none':
            xf = xf.contiguous()

        x = torch.istft(xf, self.n_fft, hop_length=self.hop_length,
                        normalized=self.normalize,
                       window=self.window)
    
        return rearrange(x, '(b c) l -> b l c', c=self.in_channels)
        

def time_to_timefreq(
        x: torch.Tensor, n_fft: int,
        C: int, norm:bool=True,
        split_type: str='real_imag',
        stft_kwargs: Optional[dict]=None
    ) -> torch.Tensor:
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


def timefreq_to_time(
        x: torch.Tensor, n_fft: int, C: int,
        split_type: str = 'real_imag', 
        norm:bool=True,
        stft_kwargs: Optional[dict]=None
    ) -> torch.Tensor:
    """
    Convert time-frequency domain tensor (STFT representation) back to the time domain.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor in the time-frequency domain.
        Shape (batch_size, channels, frequencies, length)
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
    

def plot_timefreq(xf: torch.Tensor, split_type: str):
    """
    Plot the time-frequency representation.

    Parameters
    ----------
    xf : torch.Tensor
        A tensor of shape (2, T, F) representing the complex frequency components.
    split_type : str
        The method used to split complex numbers in STFT.
        Determines how the first axis of xf is interpreted.
    """
    if xf.shape[0] != 2:
        raise ValueError(
            "xf must have shape (2, T, F), "
            "where the first axis holds two components.")
    
    _, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    im1 = axs[0].imshow(xf[0].numpy(), aspect='auto', origin='lower', cmap='jet')
    if split_type == 'real_imag':
        axs[0].set_title('Real Part of Frequency Components')
    elif split_type == 'magnitude_phase':
        axs[0].set_title('Magnitude of Frequency Components')

    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    plt.colorbar(im1, ax=axs[0], orientation='vertical')
    
    # Plot imaginary part on the second subplot
    im2 = axs[1].imshow(xf[1].numpy(), aspect='auto', origin='lower', cmap='jet')
    if split_type == 'real_imag':
        axs[1].set_title('Imaginary Part of Frequency Components')
    elif split_type == 'magnitude_phase':
        axs[1].set_title('Phase of Frequency Components')

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    plt.colorbar(im2, ax=axs[1], orientation='vertical')

    plt.tight_layout()
    plt.show()


def zero_pad_high_freq(
        xf: torch.Tensor,
        n_low_freq: int=1,
        copy: bool=False
    ) -> torch.Tensor:
    """
    Fill all frequencies other than mean energy (component 0)
    with zeroes or copies of the mean energy.

    Parameters
    ----------
    xf : torch.Tensor
        of shape (B, C, H, W);
        H: frequency-axis, W: temporal-axis
    n_low_freq : int, optional
        Number of low-frequency components to keep.
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
        xf_l = torch.zeros(xf.shape).to(xf.device) # (b c h w)
        xf_l[:, :, :n_low_freq, :] = xf[:, :, :n_low_freq, :]
    else:
        # model input: copy the highest component of LF band and 
        # paste it to the rest of the frequency components
        xf_l_part = xf[:, :, :n_low_freq, :]  # (b c n_low_freq w)
        n_high_freq = xf.shape[2] - n_low_freq
        xf_h = repeat(xf_l[:,:,[n_low_freq-1],:], 'b c 1 w -> b c h w', h=n_high_freq)
        xf_l = torch.cat((xf_l_part, xf_h), dim=2).float()

    return xf_l


def zero_pad_low_freq(
        xf: torch.Tensor,
        n_low_freq: int=1,
        copy: bool=False) -> torch.Tensor:
    """
    Fill the mean energy (component 0)
    with zeroes or a copy of component 1.

    Parameters
    ----------
    xf : torch.Tensor
        of shape (B, C, H, W);
        H: frequency-axis, W: temporal-axis
    n_low_freq : int, optional
        Number of low-frequency components to remove.
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
        xf_h[:, :, n_low_freq:, :] = xf[:, :, n_low_freq:, :]
    else:
        # model input: copy the first HF component,
        # and paste it to all LF components
        xf_h_part = xf[:, :, n_low_freq:, :]  # (b c h-1 w)
        xf_l = repeat(xf_h[:,:,[n_low_freq],:],
                      'b c 1 w -> b c h w', h=n_low_freq)
        xf_h = torch.cat((xf_l, xf_h_part), dim=2).float()

    return xf_h


def stft_decomp(x: torch.Tensor,
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


def compute_downsample_rate(
        input_length: int,
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


def quantize(z: torch.Tensor, vq_model,
             transpose_channel_length_axes: bool=False,
             svq_temp:Union[float,None]=None):
    """
    Quantise embedding z according to whether
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


### For wavelets ###
def _legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out

def _phi(phi_c, x, lb = 0, ub = 1):
    mask = np.logical_or(x<lb, x>ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

def _get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k,k))
    phi_2x_coeff = np.zeros((k,k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
            phi_coeff[ki,:ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
            phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                a = phi_2x_coeff[ki,:ki+1]
                b = phi_coeff[i, :i+1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]
            for j in range(ki):
                a = phi_2x_coeff[ki,:ki+1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            a = psi1_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm1 = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm2 = (prod_ * 1/(np.arange(len(prod_))+1) * (1-np.power(0.5, 1+np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i,:])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i,:])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i,:])) for i in range(k)]
    
    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki,:ki+1] = np.sqrt(2/np.pi)
                phi_2x_coeff[ki,:ki+1] = np.sqrt(2/np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2*x-1), x).all_coeffs()
                phi_coeff[ki,:ki+1] = np.flip(2/np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4*x-1), x).all_coeffs()
                phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                
        phi = [partial(_phi, phi_coeff[i,:]) for i in range(k)]
        
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2)* phi[ki](2*x_m)).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2*x_m)).sum()        
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            psi1[ki] = partial(_phi, psi1_coeff[ki,:], lb = 0, ub = 0.5)
            psi2[ki] = partial(_phi, psi2_coeff[ki,:], lb = 0.5, ub = 1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

            psi1[ki] = partial(_phi, psi1_coeff[ki,:], lb = 0, ub = 0.5+1e-16)
            psi2[ki] = partial(_phi, psi2_coeff[ki,:], lb = 0.5+1e-16, ub = 1)
        
    return phi, psi1, psi2


def get_filter(base: str, k: int) -> tuple:
    """
    Get the filter wavelet matrix for the given base and number of filters

    Parameters
    ----------
    base : str
        'legendre' or 'chebyshev'
        The type of base functions to use.
    k : int
        Number of filters
    
    Returns
    -------
    tuple
        Tuple of 6 numpy arrays: H0, H1, G0, G1, PHI0, PHI1
        H0, H1: High pass filter for even and odd indices
        G0, G1: Low pass filter for even and odd indices
        PHI0, PHI1: Filter for the scaling function
    """
    
    def psi(psi1, psi2, i, inp):
        mask = (inp<=0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1-mask)
    
    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')
    
    x = Symbol('x')
    H0 = np.zeros((k,k))
    H1 = np.zeros((k,k))
    G0 = np.zeros((k,k))
    G1 = np.zeros((k,k))
    PHI0 = np.zeros((k,k))
    PHI1 = np.zeros((k,k))
    phi, psi1, psi2 = _get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1/k/_legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
        
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
                
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
                
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2*x_m) * phi[kpi](2*x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2*x_m-1) * phi[kpi](2*x_m-1)).sum() * 2
                
        PHI0[np.abs(PHI0)<1e-8] = 0
        PHI1[np.abs(PHI1)<1e-8] = 0

    H0[np.abs(H0)<1e-8] = 0
    H1[np.abs(H1)<1e-8] = 0
    G0[np.abs(G0)<1e-8] = 0
    G1[np.abs(G1)<1e-8] = 0
        
    return H0, H1, G0, G1, PHI0, PHI1