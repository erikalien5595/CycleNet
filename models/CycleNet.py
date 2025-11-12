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
        # self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length, cycle_data):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return cycle_data[gather_index]
        # return self.data[gather_index]


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, d_ff, dropout):
        super(MLPBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim),
        )

    def forward(self, x):
        return x + self.model(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.use_revin = configs.use_revin
        self.t_dim = configs.t_dim
        self.s_dim = configs.s_dim
        self.use_day_index = configs.use_day_index
        self.use_hour_index = configs.use_hour_index
        self.use_min_index = configs.use_min_index
        self.emb_len_hour = configs.hour_length
        self.emb_len_day = configs.day_length
        self.emb_len_min = configs.min_length

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        self.linear_emb = nn.Linear(self.seq_len, self.d_model)

        if self.use_day_index:
            self.emb_day = nn.Parameter(torch.zeros(self.emb_len_day, self.t_dim), requires_grad=True)
        if self.use_hour_index:
            self.emb_hour = nn.Parameter(torch.zeros(self.emb_len_hour, self.t_dim), requires_grad=True)
            self.emb_month = nn.Parameter(torch.zeros(12, self.t_dim), requires_grad=True)
            self.emb_day_in_month = nn.Parameter(torch.zeros(31, self.t_dim), requires_grad=True)
        # if self.use_min_index:
        #     self.emb_min = nn.Parameter(torch.zeros(self.emb_len_min, self.t_dim), requires_grad=True)

        self.node_emb = nn.Parameter(torch.empty(self.s_dim, self.enc_in))  # s_dim, N
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.emb_day)
        nn.init.xavier_uniform_(self.emb_hour)
        nn.init.xavier_uniform_(self.emb_month)
        nn.init.xavier_uniform_(self.emb_day_in_month)
        # nn.init.xavier_uniform_(self.emb_min)

        self.input_dim = self.d_model+self.s_dim+self.t_dim
        self.output_dim = self.d_model+self.s_dim+self.t_dim
        if self.model_type == 'linear':
            self.model = nn.Linear(self.input_dim, self.output_dim)
        elif self.model_type == 'mlp':
            # self.model = nn.ModuleList([MLPBlock(self.input_dim, self.output_dim, self.d_ff, configs.dropout)
            #                             for _ in range(configs.e_layers)])
            self.model = nn.Sequential(*[MLPBlock(self.input_dim, self.output_dim, self.d_ff, configs.dropout)
                                        for _ in range(configs.e_layers)])
            # self.model = nn.Sequential(
            #     nn.Linear(self.input_dim, self.d_ff),
            #     nn.ReLU(),
            #     nn.Dropout(configs.dropout),
            #     nn.Linear(self.d_ff, self.output_dim),
            # )
        # self.regression = nn.Conv1d(
        #     in_channels=self.output_dim, out_channels=self.pred_len, kernel_size=1, bias=True)
        self.regression = nn.Linear(self.output_dim, self.pred_len)

    def forward(self, x, cycle_index, cycle_data, hour_index=None, day_index=None, month_index=None, day_in_month_index=None):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        batch_size, _, _ = x.shape
        cycle_data = torch.Tensor(cycle_data).to(cycle_index.device)
        # print('hour_index', hour_index, 'day_index', day_index)

        # # remove the cycle of the input data
        # x = x - self.cycleQueue(cycle_index, self.seq_len, cycle_data)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len, cycle_data)
        # Q = self.cycleQueue(cycle_index, self.seq_len, cycle_data)
        # Q_mean = torch.mean(Q, dim=1, keepdim=True)
        # Q_std = torch.sqrt(torch.var(Q, dim=1, keepdim=True) + 1e-5)
        # x = x - (Q - Q_mean) / Q_std  # self.cycleQueue(cycle_index, self.seq_len, cycle_data)

        x = self.linear_emb(x.permute(0, 2, 1)).permute(0, 2, 1)  # batch_size, d_model, N
        emb_day = torch.zeros(batch_size, self.t_dim, self.enc_in).to(x.device)
        emb_hour = torch.zeros(batch_size, self.t_dim, self.enc_in).to(x.device)
        emb_month = torch.zeros(batch_size, self.t_dim, self.enc_in).to(x.device)
        emb_day_in_month = torch.zeros(batch_size, self.t_dim, self.enc_in).to(x.device)
        # emb_min = torch.zeros(batch_size, self.t_dim, self.enc_in).to(x.device)
        if self.use_day_index:
            emb_day = self.emb_day[day_index.long()]
            emb_day = emb_day.unsqueeze(-1).expand(-1, -1, self.enc_in)  # batch_size, t_dim, N
        if self.use_hour_index:
            emb_hour = self.emb_hour[hour_index.long()]
            emb_hour = emb_hour.unsqueeze(-1).expand(-1, -1, self.enc_in)  # batch_size, t_dim, N
            emb_month = self.emb_month[month_index.long()]
            emb_month = emb_month.unsqueeze(-1).expand(-1, -1, self.enc_in)  # batch_size, t_dim, N
            emb_day_in_month = self.emb_day_in_month[day_in_month_index.long()]
            emb_day_in_month = emb_day_in_month.unsqueeze(-1).expand(-1, -1, self.enc_in)  # batch_size, t_dim, N
            emb_hour = emb_hour
        # if self.use_min_index:
        #     emb_min = self.emb_min[(min_index % self.emb_len_min).long()]
        #     emb_min = emb_min.unsqueeze(-1).expand(-1, -1, self.enc_in)  # batch_size, t_dim, N
        time_emb = emb_day + emb_hour + emb_month + emb_day_in_month # + emb_min

        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # batch_size, s_dim, N
        x = torch.cat([x, time_emb, node_emb], dim=1)  # batch_size, d_model+t_dim+s_dim, N
        # forecasting with channel independence (parameters-sharing)
        x = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        y = self.regression(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len, cycle_data)
        # Q1 = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len, cycle_data)
        # Q_mean1 = torch.mean(Q1, dim=1, keepdim=True)
        # Q_std1 = torch.sqrt(torch.var(Q1, dim=1, keepdim=True) + 1e-5)
        # y = y + (Q1 - Q_mean1) / Q_std1

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
