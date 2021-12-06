import torch.nn as nn 

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):

    out_channel = 32
    hidden_size = 64
    kernel_size = 3
    stride = 1
    use_Relu = True
    use_orthogonal = True
    
    def __init__(self, state_shape):
        
        super().__init__()
        
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_ReLU])
        input_channel = 1
        input_width = state_shape[0]
        input_height = state_shape[1]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        cnn_out_size = self.out_channel // 2 * (input_width - self.kernel_size + self.stride) * (input_height - self.kernel_size + self.stride)
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(cnn_out_size,
                            self.hidden_size)),
                            active_func,
                            )
    def forward(self, input):
        output = self.cnn(input)
        return output