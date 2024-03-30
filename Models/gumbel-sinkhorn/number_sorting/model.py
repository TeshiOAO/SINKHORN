import torch
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, hid_c, out_c):
        super().__init__()

        self.g1 = nn.Sequential(
            nn.Conv1d(1, hid_c, 1),
            nn.ReLU(True),
            nn.Conv1d(hid_c, hid_c*2, 1),
            nn.ReLU(True),
            nn.Conv1d(hid_c*2, hid_c*4, 1),
            nn.ReLU(True),
        )

        self.g2 = nn.Sequential(
            nn.Conv1d(hid_c*4, hid_c*2, 1),
            nn.Conv1d(hid_c*2, hid_c, 1),
            nn.Conv1d(hid_c, out_c, 1),
        )

        self.out_c = out_c
        self.positions = nn.Embedding(out_c, out_c//3+1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, xyz):
        # NOTE: x is 2-dim or 3-dim torch.Tensor, i.e., (batch, 1, numbers)
        pos_emb = self.positions(xyz).view(x.shape[0], 1, (self.out_c//3+1) *3)[:, :, :self.out_c]
        if x.dim() == 2:
            X = x[:, None].detach().clone()
        else:
            X = x.detach().clone()
        X += pos_emb
        h = self.g1(X)
        log_alpha = self.g2(h).transpose(1,2).contiguous()
        # pod_log_alpha = log_alpha + pos_emb
        return log_alpha