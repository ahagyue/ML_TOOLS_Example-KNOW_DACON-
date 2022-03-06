import torch
import torch.nn as nn

from .base_config import BaseConfig

from typing import Dict, Any

class ExpConfig(BaseConfig):
    _hyperparam : Dict[str, Any]

    def __init__(self):
        super(ExpConfig, self).__init__()

        self._hyperparam = {}

        self._hyperparam['layers'] = [
            [77, n, n, 538] for n in range(500, 1000, 100)
        ]
        self._hyperparam['batch_size'] = {
            'values': [4, 8, 16]
        }
        self._hyperparam['learning_rate'] = {
            'values': [1e-4]
        }
        self._hyperparam['weight_decay'] = {
            'values': [5e-6]
        }
        self._hyperparam['criterion'] = {
            'values': [nn.CrossEntropyLoss()]
        }
        self._hyperparam['optimizer'] = {
            'values': [
                lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd)
            ]
        }

        GPU_NUM = 1
        self._args['device'] = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        self._args['early_stopping_step'] = 5

    def get_exp_arg(self, model=None) -> Dict:
        if model:
            self._hyperparam['model'] = {
                'values': [model(layer) for layer in self._hyperparam['layers']]
            }
            del self._hyperparam['layers']
        exp_arg = {
            'method': 'grid',  # grid, random
            'metric': {
                'name': 'Test_Loss',
                'goal': 'minimize'
            },
            'parameters': self._hyperparam
        }
        return exp_arg

    def get_default_arg(self, model=None) -> Dict:
        default_arg: Dict[str, Any] = {}
        keys = self._hyperparam.keys()

        for key in keys:
            default_arg[key] = self._hyperparam[key]['values'][0]
        return default_arg