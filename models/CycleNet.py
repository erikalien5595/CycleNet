import torch
import torch.nn as nn

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
        self.data = torch.nn.Parameter(torch.randn((cycle_len, channel_size)), requires_grad=True)

    def forward(self, index, length):
        # print(index.view(-1, 1))
        # print(torch.arange(length, device=index.device).view(1, -1))
        # print(index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1))
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        # print(gather_index, gather_index.shape)
        # print(self.data[gather_index].shape)
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

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, x_dec, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # forecasting with channel independence (parameters-sharing)
        trend_pred = y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        # trend_true = x_dec[-self.pred_len:] - self.cycleQueue(cycle_index, self.seq_len+self.pred_len)[:, -self.pred_len:, :]
        trend_true = x_dec[:, -self.pred_len:, :] - self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
            trend_pred = trend_pred * torch.sqrt(seq_var) + seq_mean
            trend_true = trend_true * torch.sqrt(seq_var) + seq_mean
            # print(trend_true-trend_pred)
        return y, trend_pred, trend_true

if __name__=='__main__':
    cycle_len = 6
    seq_len = 11
    pred_len = 5
    cycle_index = torch.Tensor((10, )).int()
    cycle = RecurrentCycle(cycle_len, 3)
    print(cycle(cycle_index, seq_len))
    print(cycle(cycle_index, seq_len+pred_len)[:, -pred_len:, :])
    print(cycle((cycle_index + seq_len) % cycle_len, pred_len))
    torch.utils.data.Dataset

