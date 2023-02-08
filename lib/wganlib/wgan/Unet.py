from absl import flags
import torch.nn as nn
import torch.optim.lr_scheduler

flags.DEFINE_float('learning_rate', 
                    0.001,
                    'set the learning rate {float}.')

flags.DEFINE_integer('batch_size', 
                    64,
                    'set the batch size {integer}.')

flags.DEFINE_integer('n_epochs', 
                    50,
                    'set the number of epochs {integer}.')

flags.DEFINE_float('schedFactor', 
                    0.1, 
                    'set the factor of the scheduler {float}.')

flags.DEFINE_integer('schedPatience', 
                    3, 
                    'set the patience of the scheduler {integer}.')

flags.DEFINE_float('weight_decay', 
                    0.14, 
                    'set the weight decay {float}.')

flags.DEFINE_boolean('printShape', 
                    False, 
                    'set if the shape of the layers are being printed {boolean}.')

flags.DEFINE_integer('nchan', 
                     64,
                     'set a number of channels (feature maps) for the Generator. The number of channels are doubled as we go deeper to the Encoder of the Unet. On the other hand the number of channels are halved as we go up to the Decoder of the Unet. The last layer is hardcoded to 393.') 

flags.DEFINE_integer('ndf', 
                     64,
                     'set a number of channels (feature maps) for the Discriminator. The number of channels are doubled as we go deeper to the Encoder of the Unet. On the other hand the number of channels are halved as we go up to the Decoder of the Unet. The last layer is hardcoded to 393.') 

flags.DEFINE_integer('depth', 
                     2, 
                     'set a number of the Unet\'s depth.') 

flags.DEFINE_string('load_model',
                    '',
                    'Set a path to a model. default: \'\'')

flags.DEFINE_string('o_name',
                    '',
                    'Set a name for output denoised images. default: \'\'')

flags.DEFINE_boolean('retain_shape', 
                    True,
                    'set if the shape of the output resolution of the Unet will be the same as input.')

class UnetBottleNeck(nn.Module):
    def __init__(self, in_channels, inner_nch, outer_nch, retain_shape_to_output = True, name="1", printForward = True, bn = True):
        super(UnetBottleNeck, self).__init__()
        self.name = "unet_bottleneck_"+str(name)
        self.printShape = printForward
        if bn:
            """
            self.conv1 = ConvInstNorm1d(in_channels = in_channels,
                                  out_channels = inner_nch,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = 1 if retain_shape_to_output else 0)
            """
            self.conv1 = ConvBn1d(in_channels=in_channels,
                                        out_channels=inner_nch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1 if retain_shape_to_output else 0)
        else:
            self.conv1 = nn.Conv1d(in_channels = in_channels, 
                                   out_channels = inner_nch, 
                                   kernel_size = 3,
                                   stride = 1,
                                   padding = 1 if retain_shape_to_output else 0)
        self.func1 = nn.ReLU(True)
        if bn:
            """
            self.conv2 = ConvInstNorm1d(in_channels = inner_nch,
                                  out_channels = outer_nch,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = 1 if retain_shape_to_output else 0)
            """
            self.conv2 = ConvBn1d(in_channels=inner_nch,
                                  out_channels=outer_nch,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1 if retain_shape_to_output else 0)
        else:
            self.conv2 = nn.Conv1d(in_channels = inner_nch, 
                                   out_channels = outer_nch, 
                                   kernel_size = 3,
                                   stride = 1,
                                   padding = 1 if retain_shape_to_output else 0)
        self.func2 = nn.ReLU(True)

    def forward(self, x):
        x = self.func1(self.conv1(x))
        x = self.func2(self.conv2(x))
        return x

class ConvInstNorm1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, name="1", printForward=False):
        super(ConvInstNorm1d, self).__init__()
        self.name = "conv_instnorm_1d_"+str(name)
        self.conv = nn.Conv1d(in_channels = in_channels, 
                              out_channels = out_channels, 
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              dilation = dilation)
        self.instnorm = nn.InstanceNorm1d(num_features = out_channels, track_running_stats=False)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.instnorm(out)
        return out

class ConvBn1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, name="1", printForward=False):
        super(ConvBn1d, self).__init__()
        self.name = "conv_bn_1d_"+str(name)
        self.conv = nn.Conv1d(in_channels = in_channels, 
                              out_channels = out_channels, 
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              dilation = dilation)
        self.bn = nn.BatchNorm1d(num_features = out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

def make_encoder_layers_from_hps(hps, retain_shape_to_output = True):
    """ Create the layers of the Unet.
        - hps:
            -- depth: deepest Unet bottleneck. It is placed after the encoder and before the decoder.
            -- nchan: number of channels of the "first bottleneck". "First Bottleneck" is considered the first bottleneck of the Unet. 
                         As we go deeper to the encoder the number of channels doubles.
            -- 
    """

    # the deepest bottleneck consists of: initial channels * 2 ^ deepest BottleNeck.
    model = UnetBottleNeck(in_channels = hps['nchan']*2**(hps['depth']-1),
                           inner_nch = hps['nchan']*2**hps['depth'],
                           outer_nch = hps['nchan']*2**(hps['depth']-1),
                           retain_shape_to_output = retain_shape_to_output,
                           name = str(hps['depth'])) 
    
    # Build bottom-up layer-by-layer! 
    for i in reversed(range(hps['depth'])):
        encoder = UnetBottleNeck(in_channels = int(hps['nchan']*2**(i-1)),
                                 inner_nch = int(hps['nchan']*2**i),
                                 outer_nch = int(hps['nchan']*2**i),
                                 retain_shape_to_output = retain_shape_to_output,
                                 name = "encoder_"+str(i)) 
        down = nn.MaxPool1d(kernel_size = 2,
                            stride = 2
                            )
        up = nn.ConvTranspose1d(in_channels = int(hps['nchan']*2**i),
                                out_channels = int(hps['nchan']*2**i),
                                kernel_size = 2 if i!= 0 else 3,
                                stride = 2,
                                padding = 0
                                )
        decoder = UnetBottleNeck(in_channels = int(hps['nchan']*2**(i+1)),
                                 inner_nch = int(hps['nchan']*2**i),
                                 outer_nch = int(hps['nchan']*2**(i-1)),
                                 retain_shape_to_output = retain_shape_to_output,
                                 name = "decoder_"+str(i))
        if i == 0:
            encoder = UnetBottleNeck(in_channels = hps['nc'],
                                     inner_nch = int(hps['nchan']),
                                     outer_nch = int(hps['nchan']),
                                     retain_shape_to_output = retain_shape_to_output,
                                     name = "encoder_"+str(i)
                                     )
            decoder = UnetBottleNeck(in_channels = int(hps['nchan']*2**(i+1)),
                                     inner_nch = int(hps['nchan']*2**i),
                                     outer_nch = int(hps['nchan']*2**i),
                                     retain_shape_to_output = retain_shape_to_output,
                                     name = "decoder_"+str(i)
                                     )
            outermost = nn.Conv1d(in_channels = int(hps['nchan']*2**i),
                                  out_channels = hps['nc'],
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0)
            model = [encoder] + [down] + [model] + [up] + [decoder] + [outermost]
        else:    
            model = [encoder] + [down] + [model] + [up] + [decoder]

    return model

class Unet1d(nn.Module):
    """Neural Network module.
        Edit architecture in __init__() and forward().
        --- x_channels:
        --- name:
    """
    def __init__(self, hps, printForward = False, retain_shape_to_output = True):
        super(Unet1d, self).__init__()
        self.name = "Unet"
        self.printShape = printForward
        self.hps = hps
        
        self.layers = make_encoder_layers_from_hps(hps, retain_shape_to_output)
        #print(self.layers, '\n')
        self.__assign_layers_to_self()

    # I need to fix that!
    def __assign_layers_to_self(self):
         
        def assign_layers(self, layers, i):
            if isinstance(layers, list):
                for layer in layers:
                    i = assign_layers(self, layer, i)
            else:
                layer_name = f"unet_layer_{i}"
                setattr(self, layer_name, layers)
                #print(f"{i} - {layers}")
                i += 1
            return i

        i = 0
        assign_layers(self, self.layers, i)

    def get_embedding(self, x):
        i = 0
        cropped = []
        layer_name = lambda x: f"unet_layer_{x}"
        while hasattr(self, layer_name(i)):
            layer = getattr(self, layer_name(i))
            # 2*depth: number of layers of the encoder. 2 -> Bottleneck and MaxPool.
            if i < 2*self.hps['depth'] and i%2 == 0: 
                x = layer(x)
                #print(f'UNET {i}: {x.shape}')
                cropped.append(x)
            elif i > 2*self.hps['depth'] and i%2 == 0:
                paste = torch.cat([cropped.pop(-1)[:x.shape[0], :x.shape[1], :x.shape[2]], x], axis=1)
                x = layer(paste)
                #print(f'UNET {i}: {x.shape}')
            else:
                x = layer(x)
                #print(f'UNET {i}: {x.shape}')
            i += 1
        return x

    def forward(self, x):
        #print('#################### Forward Passing ####################')
        out = self.get_embedding(x)
        return out


def get_criterion_from_hps(hps):
    if hps['criterion'] == 'GaussianNLLLoss':
        return nn.GaussianNLLLoss()
    elif hps['criterion'] == 'MSELoss':
        return nn.MSELoss()
    else:
        raise Exception("NYI")

def get_optimizer_from_hps(hps, net):
    if hps['optimizer'] == 'Adam': 
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=hps['lr'],
                                     weight_decay=hps['weight_decay'])
    elif hps['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=hps['lr'],
                                    weight_decay=hps['weight_decay'])
    else:
        raise Exception("NYI")
    return optimizer

