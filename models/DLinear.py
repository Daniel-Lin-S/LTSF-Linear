import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size: int, stride: int):
        """
        Parameters
        ----------
        kernel_size: int
            the size of the averaging window
        stride: int
            distance between starting point
            of adjacent windows. 
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Compute a smooth version of x

        Parameter
        ---------
        x : torch.Tensor
            of shape (batch_size, length, channels)

        Return
        ------
        torch.Tensor
            of the same shape as input
        """
        # padding on the both ends of time series to preserve length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # take the average on temporal axis
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear Model for Time Series Forecasting.
    
    This model decomposes the input time series into seasonal and trend components, then 
    applies linear regression layers to predict the seasonal and trend components separately. 
    Finally, the predictions for both components are summed to produce the final forecast.
    
    Attributes:
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction horizon.
        decompsition (series_decomp): Decomposition method to
          separate seasonal and trend components.
        individual (bool): Whether to apply individual linear models for each channel.
        channels (int): Number of channels in the input data.
        Linear_Seasonal (nn.ModuleList or nn.Linear): Linear layer(s) for
          modeling the seasonal component.
        Linear_Trend (nn.ModuleList or nn.Linear): Linear layer(s) for
          modeling the trend component.
    """
    def __init__(self, configs):
        """
        Parameters
        ----------
        configs : object
            Configuration object that contains the following attributes:
            
            - `seq_len` (int): Length of the input sequence.
            - `pred_len` (int): Length of the prediction horizon.
            - `individual` (bool): Whether to use separate linear
              models for each input channel.
            - `enc_in` (int): Number of input channels.
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        """
        Forward pass of the model. The input time series is decomposed into seasonal 
        and trend components, then passed through linear models to predict the seasonal 
        and trend components. The final prediction is the sum of these components.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, seq_len, channels)

        Returns
        -------
        torch.Tensor
            The predicted values with shape (batch_size, pred_len, channels).
        """
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual: # separate linear model for each channel 
            # prepare empty torch tensors
            seasonal_output = torch.zeros(
                [seasonal_init.size(0),seasonal_init.size(1),self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0),trend_init.size(1),self.pred_len],
                dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else: # shared linear model for all channels
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
