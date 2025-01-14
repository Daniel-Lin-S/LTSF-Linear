import os
import torch
from utils.tools import format_large_int
from utils.logger import Logger
from abc import ABC, abstractmethod
from typing import Optional


class Exp_Basic(ABC):
    """
    Base class for experiments
    """
    def __init__(self, args,
                 logger: Optional[Logger]=None):
        self.args = args
        self.logger = logger
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    @abstractmethod
    def _get_data(self):
        pass

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError
    
    def _print_model(self, model: torch.nn.Module, model_name: str):
        total_params = sum(p.numel() 
                           for p in model.parameters() 
                           if p.requires_grad)
        total_static_params = sum(p.numel() for p in model.parameters()
                                  if not p.requires_grad)
        message = (
            f"------ Summary of Model {model_name} --------\n"
            f"Number of trainable parameters of the model {model_name}: "
            f"{format_large_int(total_params)}\n"
            f"Number of static parameters of the model {model_name}: "
            f"{format_large_int(total_static_params)}. "
        )
        if self.logger:
            self.logger.log(message, level='info')
        else:
            print(message)

    def _acquire_device(self):
        if self.args.use_gpu and self.args.use_multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            device = torch.device('cuda')  # use all visible devices
            message = f'Using multiple GPUs: {self.args.devices}'
        elif self.args.use_gpu: # single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            message = f'Using single GPU: cuda:{self.args.gpu}'
        else:
            device = torch.device('cpu')
            message = 'Using CPU'

        if self.logger:
            self.logger.log(message)
        else:
            print(message)

        return device

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
