import torch
import torch.nn as nn

class KnowNet(nn.Module):
    def __init__(self, group, layers : list, amp_factor=2):
        super(KnowNet, self).__init__()

        self.group = group
        self.cum_group = [0]
        self.small_net_list = []

        for n in group:
            if n != 1:
                self.small_net_list.append(
                    nn.Sequential(
                        nn.Linear(n, n * amp_factor),
                        nn.ReLU(),
                        nn.Linear(n * amp_factor, 1),
                        nn.ReLU()
                    )
                )
            self.cum_group.append(self.cum_group[-1] + n)
        self.small_net_list = nn.ModuleList(self.small_net_list)

        layer = []
        for i in range(len(layers) - 1):
            layer.append(nn.Linear(layers[i], layers[i + 1]))
            layer.append(nn.ReLU())
        layer.pop()
        self.linear_layer = nn.Sequential(*layer)

    def forward(self, x):
        _x = []
        nth_net = 0
        for i in range(len(self.group)):
            if self.cum_group[i+1] - self.cum_group[i] <= 1:
                sx = x[:, self.cum_group[i] : self.cum_group[i+1]]
            else:
                sx = self.small_net_list[nth_net](x[:, self.cum_group[i] : self.cum_group[i+1]])
                nth_net += 1
            _x.append(sx)
        x = torch.cat(_x, dim=1)

        return self.linear_layer(x)

