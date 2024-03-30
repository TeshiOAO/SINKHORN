import numpy as np
import torch
import torch.nn as nn

# trunc_normal = nn.initializer.TruncatedNormal(std=.02)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.device = "cpu"

    def forward(self, batch, xyz):
        """
        :param tensor: A 5d tensor of size (batch_size, ch)
        :return: Positional Encoding Matrix of size (batch_size, ch)
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        self.cached_penc = torch.zeros((batch, self.org_channels), device=self.device, dtype=self.inv_freq.dtype,)
        sin_inp_x = torch.einsum("i,j->ij", x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", z, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros( # int(x.item()), int(y.item()), int(z.item()),
            (batch, self.channels * 3),
            device=self.device,
            dtype=self.inv_freq.dtype,
        )
        emb[:, : self.channels] = emb_x
        emb[:, self.channels : 2 * self.channels] = emb_y
        emb[:, 2 * self.channels :] = emb_z
        self.cached_penc = emb[:, :self.org_channels]

        return self.cached_penc


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor, batch, xyz):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(batch, xyz)
        penc = penc.to(tensor.device)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size(), penc.size()
        )
        return tensor + penc


def learnable_embedding(pos_len, dim):
    pos_emb = nn.Embedding(pos_len, dim)
    # nn.init.constant_(pos_emb.weight, 0)
    
    return pos_emb
