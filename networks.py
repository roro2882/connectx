import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCQ(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[32,32],
                 activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc
        self.input_dim = input_dim
        out_channels = 64

        self.input_layer = nn.Conv2d(input_dim[0], out_channels = out_channels, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.input_layer1 = nn.Conv2d(out_channels, out_channels = out_channels, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.pool_layer    = nn.MaxPool2d((input_dim[1],input_dim[2]))
        #self.dropout = nn.Dropout2d(0.5)

        self.hidden_layers = nn.ModuleList()
        hidden_dims = list(hidden_dims)
        hidden_dims[0] = (input_dim[1])*(input_dim[2])*out_channels
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1]+1, output_dim)
        self.bias_layer= nn.Linear(out_channels, 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            if len(x.shape)==len(self.input_dim):
                x = x.unsqueeze(0)
        return x

    def forward(self, state, drop =False):
        x = self._format(state)
        x = self.input_layer(x)
        x = self.activation_fc(x)
        x = self.input_layer1(x)
        x = self.activation_fc(x)
        y = self.pool_layer(x)
        bias = self.bias_layer(y.flatten(start_dim=1))
        bias = self.activation_fc(bias)
        if drop:
            pass
        if drop:
            pass
        x = torch.flatten(x,start_dim=1)
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = torch.concat((x,bias),dim=1)
        x = self.output_layer(x)
        return x

    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals

