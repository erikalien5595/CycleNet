import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        # trend-residual decomposition
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model_trend = nn.Linear(self.seq_len, self.pred_len)
            self.model_resid = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model_trend = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )
            self.model_resid = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )


    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # remove the trend from the remainder of the last step
        resid, trend = self.decomp(x)

        # forecasting with channel independence (parameters-sharing)
        y_trend = self.model_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        y_resid = self.model_trend(resid.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y_trend + y_resid + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            # De-Normalization
            y = y * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            y = y + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            # y = y * seq_std + seq_mean

        return y
