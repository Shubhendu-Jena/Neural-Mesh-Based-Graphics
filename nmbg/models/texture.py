import torch
import torch.nn as nn



class Texture(nn.Module):
    def null_grad(self):
        raise NotImplementedError()

    def reg_loss(self):
        return 0.


class PointTexture(Texture):
    def __init__(self, num_channels, size, activation='none', checkpoint=None, init_method='zeros', reg_weight=0.):
        super().__init__()

        assert isinstance(size, int), 'size must be int'

        shape = 1, num_channels, size

        if checkpoint:
            self.texture_ = torch.load(checkpoint, map_location='cpu')['texture'].texture_
        else:
            if init_method == 'rand':
                texture = torch.rand(shape)
            elif init_method == 'zeros':
                texture = torch.zeros(shape)
            else:
                raise ValueError(init_method)
            self.texture_ = nn.Parameter(texture.float())

        self.activation = activation
        self.reg_weight = reg_weight

    def null_grad(self):
        self.texture_.grad = None

    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.texture_, 2))
        