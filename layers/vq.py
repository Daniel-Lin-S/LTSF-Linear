"""
The code is taken from https://github.com/lucidrains/vector-quantize-pytorch
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast
from torch.distributions.categorical import Categorical

from einops import rearrange, repeat
from typing import Optional, Union

from utils import *


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def softmax_sample(t, temperature, dim=-1):
    if isinstance(temperature, type(None)):
        return t.argmax(dim=dim)

    m = Categorical(logits=t / temperature)
    return m.sample()


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


# regularization losses

def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


# distance types

class EuclideanCodebook(nn.Module):
    """
    source: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    """
    def __init__(
            self,
            dim,
            codebook_size,
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0,
            emb_dropout=0.,
    ):
        """
        :param dim: dimension of each code word
        :param codebook_size: number of discrete embeddings
        :param kmeans_init: if true, initialised by K-mean, otherwise initialsed randomly
        :param kmeans_iters: number of K-means iterations
        :param decay: Exponential moving average (EMA) decay rate
            for updating the codebook during training.
        :param eps: added to cluster size when updating codebook
            to prevents division by 0 cluster sizes.
        :param threshold_ema_dead_code: threshold on
            number of times a code word is used.
            If lower than this, will be replaced.
        :param learnable_codebook: if True, codebook will be updated during training
            otherwise, it relies on EMA for updating.
        :param emb_dropout: drop out rate on the codebook.
            useful for training when codebooks are learnable
        """
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.emb_dropout = emb_dropout

        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        self.embed_onehot = None
        self.perplexity = None

    @torch.jit.ignore
    def init_embed_(self, data):
        """initialise codebook"""
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        """
        replace dead words by simply re-sampling a vector
        from the samples.

        :param samples: the pool of samples
        :param mask: specifies indices which the codebook
            should be replaced.
        """
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        """
        Replace the dead code words (only used few times)
        """
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, svq_temp:Union[float,None]=None):
        """
        :param x: torch.Tensor
            with more than 2 dimensions.
            The last axis will be embedded
        :param sqv_temp: used when assigning codeword
            Draw the codeword weighted by Euclidean distance,
            higher temperature gives more uniform distribution. \n
            if None, deterinistically chooses the closest codeword.
        """
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = embed.t()

        if self.emb_dropout and self.training:
            embed = F.dropout(embed, self.emb_dropout)

        dist = -(
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ embed
                + embed.pow(2).sum(0, keepdim=True)
        )
        # embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        temp = svq_temp
        if self.training:
            embed_ind = softmax_sample(dist, dim=-1, temperature=temp)
        else:
            embed_ind = softmax_sample(dist, dim=-1, temperature=temp)  # no stochasticity
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        # recovers the original shape of x
        embed_ind = embed_ind.view(*shape[:-1])
        # fills in the code words
        quantize = F.embedding(embed_ind, self.embed)

        # apply EMA with Laplace smoothing to ensure
        # no abrupt change in the codebook when updating
        if self.training:
            cluster_size = embed_onehot.sum(0)
            # synchronise across all devices
            self.all_reduce_fn(cluster_size)

            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        # calculate perplexity score
        avg_probs = torch.mean(embed_onehot, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.embed_onehot = embed_onehot.detach()  # .cpu()
        self.perplexity = perplexity.detach()  # .cpu()

        return quantize, embed_ind


# main class
class VectorQuantize(nn.Module):
    """
    Build a codebook and quantize input.

    This class includes optional support for multi-headed quantization,
    input and output projections, and orthogonal regularization
    for codebook entries. \n
    It can process both standard feature 
    maps and image-like inputs.
    """
    def __init__(
            self,
            dim: int,
            codebook_size: int,
            codebook_class: nn.Module=EuclideanCodebook,
            codebook_params: dict={},
            heads: int=1,
            sync_codebook: bool=False,
            commitment_weight: float=1.,
            orthogonal_reg_weight: float=0.,
            orthogonal_reg_active_codes_only: bool=False,
            orthogonal_reg_max_codes: Optional[int]=None,
            **kwargs
    ):
        """
        Initialises the VectorQuantize module.

        Parameters
        ----------
        dim : int
            The dimensionality of the input feature vectors.
        codebook_size : int
            The number of codewords in the codebook.
        codebook_class : nn.Module, optional
            The class to use for the codebook. \n
            It msut have a forward method
            taking a tensor and a float (sampling temperature),
            and has a perplexity attribute.
            Default is EuclideanCodebook.
        codebook_params : dict, optional
            A dictionary containing parameters related
            to codebook construction.
            - set codebook_dim if code words should have
            different dimensions than dim (input)
        heads : int, optional
            Number of independent codebooks to use
            (multi-headed quantization). \n
            Dimensions of the last axis will be coded
            in groups. \n
            Default is 1.
        sync_codebook : bool, optional
            Whether to synchronize the codebook across
            devices in distributed training. \n
            Default is False.
        commitment_weight : float, optional
            Weight for the commitment loss. \n
            Default is 1.0.
        orthogonal_reg_weight : float, optional
            Weight for the orthogonal regularization loss. \n
            Default is 0.0 (no regularization)
        orthogonal_reg_active_codes_only : bool, optional
            Whether to calculate orthogonal loss only
            for the active (used) codewords. \n
            Default is False.
        orthogonal_reg_max_codes : int, optional
            Maximum number of codewords to use for
            orthogonal loss calculation. \n
            Default is None.
        """
        super().__init__()

        codebook_dim = codebook_params.get("codebook_dim", dim)
        codebook_input_dim = codebook_dim * heads
        self.codebook_size = codebook_size
        self.input_dim = dim

        # projection settings
        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if (
            requires_projection) else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if (
            requires_projection) else nn.Identity()

        # loss related parameters
        self.commitment_weight = commitment_weight
        codebook_params['learnable_codebook'] = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        # distributed computing parameters
        self.heads = heads
        codebook_params['use_ddp'] = sync_codebook

        self._codebook = codebook_class(
            dim=codebook_dim,
            codebook_size=codebook_size,
            **codebook_params
        )

    @property
    def codebook(self):
        return self._codebook.embed

    def forward(self, x: torch.Tensor, svq_temp:Union[float,None]=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            of shape (B N D) if accept_image_fmap=False,
            codewords will be assigned to the third dimension
            i.e. each point (b, n, ) gets a codeword
        sqv_temp : float, optional
            used when assigning codeword.
            Draw the codeword weighted by Euclidean distance,
            higher temperature gives more uniform distribution. \n
            if None, deterinistically chooses the closest codeword.

        Returns
        -------
        quantize : torch.Tensor
            The quantized tensor where each feature vector is
            replaced with the closest codeword.
            shape: (B N D) (same as input)
        embed_ind : torch.Tensor
            Indices of the selected codewords in the codebook,
            with one index per input feature vector.
        vq_loss : dict
            A dictionary containing the following keys:
            - `commit_loss`: The commitment loss, 
            encouraging inputs to stay close to the selected codewords.
            - `orthogonal_reg_loss`: The orthogonal regularization loss, 
            encouraging diversity in codebook entries.
            - `loss`: Total quantization loss, weighted sum of two lossees.
        perplexity : float
            A measure of codebook usage diversity,
            reflecting how well the codewords are used.

        Raises
        ------
        ValueError
            If the input tensor shape is not compatible
            with the expected input format
            (e.g., incorrect dimensions).

        Notes
        -----
        - During training, the `quantize` tensor is adjusted to allow gradients
        to flow through the original input (`x`) 
        while keeping the quantized representation fixed.
        - Orthogonal regularization is applied to the codebook entries
        only when `orthogonal_reg_weight > 0`.
        """
        ### arrange the shape ###
        device, heads, is_multiheaded = x.device, self.heads, self.heads > 1
        shape = x.shape   # store original shape
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f'the input x has {x.shape[-1]} dimensions in the last axis'
                f', while the dimension set by init is {self.input_dim}. '
                'Either check input x or create a new instance'
                f' with parameter dim={x.shape[-1]}')

        if x.ndim != 3:
            raise ValueError(
                'tensor x must have at 3 axes: '
                '(batch_size, spatial, hidden_dim)'
                f'got shape {shape}')

        vq_loss = {'loss': torch.tensor([0.], device=device, requires_grad=self.training),
                   'commit_loss': 0.,
                   'orthogonal_reg_loss': 0.,
        }

        ### Projection for dimension matching ###
        x = self.project_in(x)

        if is_multiheaded:
            if x.shape[-1] % heads != 0:
                raise ValueError(
                    f'heads {heads} must be divisible by the'
                    ' dimension of the axis being embedded: '
                    f'{x.shape[-1]}. \n'
                    f'Shape of original input: {shape}'
                )
            x = rearrange(x, 'b n (h d) -> (b h) n d', h=heads)

        ### Find codewords and indices ###
        quantize, embed_ind = self._codebook(x, svq_temp)

        ### Compute losses ###
        if self.training:
            # allows `z`-part to be trainable while `z_q`-part is un-trainable.
            # `z_q` is updated by the EMA in Codebook class.
            quantize = x + (quantize - x).detach()  

            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                vq_loss['commit_loss'] = torch.tensor(commit_loss)
                vq_loss['orthogonal_reg_loss'] = torch.tensor(0.0)
                vq_loss['loss'] = vq_loss['loss'] + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the 
                    # activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and (
                    num_codes > self.orthogonal_reg_max_codes):
                    # randomly pick codes to calculate loss
                    rand_ids = torch.randperm(
                        num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                vq_loss['orthogonal_reg_loss'] = torch.Tensor(orthogonal_reg_loss)
                vq_loss['loss'] = vq_loss['loss'] + (
                    orthogonal_reg_loss * self.orthogonal_reg_weight)

        ### Restore original shape ###
        if is_multiheaded:
            quantize = rearrange(quantize, '(b h) n d -> b n (h d)', h=heads)
            embed_ind = rearrange(embed_ind, '(b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)

        return quantize, embed_ind, vq_loss, self._codebook.perplexity
