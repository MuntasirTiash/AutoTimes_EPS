import torch.nn as nn

class MLP(nn.Module):
    """Multilayer perceptron to encode/decode high dimension representation of
    sequential data.

    Args:
        f_in (int): The number of input features.
        f_out (int): The number of output features.
        hidden_dim (int, optional): The number of hidden units in the MLP.
            Defaults to 256.
        hidden_layers (int, optional): The number of hidden layers in the MLP.
            Defaults to 2.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        activation (str, optional): The activation function to use.
            One of 'relu', 'tanh', or 'gelu'. Defaults to 'tanh'.
    """
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
