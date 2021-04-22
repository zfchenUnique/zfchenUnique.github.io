from .mac_network import MACNetwork
from torch import nn
#from torchpie.config import config
from config import config


def get_model(arch: str, n_vocab: int, classes: int) -> nn.Module:
    if arch == 'mac':
        model = MACNetwork(
            n_vocab,
            classes,
            in_channels=config.getint('mac', 'in_channels'),
            dim=config.getint('mac', 'dim'),
            net_length=config.getint('mac', 'net_length'),
            embedding_dim=config.getint('mac' , 'embedding_dim'),
            self_attention=config.getboolean('mac', 'self_attention'),
            memory_gate=config.getboolean('mac', 'memory_gate'),
            dropout=config.getfloat('mac', 'dropout'),
        )

            #glove_path=config.get('mac', 'glove_path')
        return model

    else:
        raise Exception
