import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *
import torchdeepretina.utils as tdrutils
import numpy as np
from scipy import signal

DEVICE = torch.device('cuda:0')

def try_kwarg(kwargs, key, default):
    try:
        return kwargs[key]
    except:
        return default

class RetVidModel(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, chans=[8,8],
                                                bn_moment=.01, softplus=True, 
                                                inference_exp=False, img_shape=(40,50,50), 
                                                ksizes=(15,11), recurrent=False, 
                                                kinetic=False, convgc=False, 
                                                centers=None, bnorm_d=1, c_size=256,
                                                **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans 
        self.softplus = softplus 
        self.infr_exp = inference_exp 
        self.bias = bias 
        self.img_shape = img_shape 
        self.ksizes = ksizes 
        self.linear_bias = linear_bias 
        self.noise = noise 
        self.bn_moment = bn_moment 
        self.recurrent = recurrent
        self.kinetic = kinetic
        self.convgc = convgc
        self.centers = centers
        assert bnorm_d == 1 or bnorm_d == 2,\
                                "Only 1 and 2 dimensional batchnorm are currently supported"
        self.bnorm_d = bnorm_d
        self.c_size = c_size
    
    def forward(self, x):
        return x

    def extra_repr(self):
        try:
            s = 'n_units={}, noise={}, bias={}, linear_bias={}, chans={}, bn_moment={}, '+\
                                    'softplus={}, inference_exp={}, img_shape={}, ksizes={}'
            return s.format(self.n_units, self.noise, self.bias, self.linear_bias,
                                        self.chans, self.bn_moment, self.softplus,
                                        self.inference_exp, self.img_shape, self.ksizes)
        except:
            pass
    
    def requires_grad(self, state):
        for p in self.parameters():
            try:
                p.requires_grad = state
            except:
                pass

class LinearStackedBNCNN(RetVidModel):
    def __init__(self, drop_p=0, one2one=False, stack_ksizes=[3,3], stack_chans=[None,None],
                                                 final_bias=False, paddings=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'StackedNet'
        self.drop_p = drop_p
        self.one2one = one2one
        self.stack_ksizes = stack_ksizes
        self.stack_chans = stack_chans
        self.paddings = [0 for x in stack_ksizes] if paddings is None else paddings
        self.final_bias = final_bias
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        modules = []

        ##### First Layer
        # Convolution
        if one2one:
            modules.append(OneToOneLinearStackedConv2d(self.img_shape[0],self.chans[0],
                                                            kernel_size=self.ksizes[0], 
                                                            padding=self.paddings[0],
                                                            bias=self.bias))
        else:
            modules.append(LinearStackedConv2d(self.img_shape[0],self.chans[0],
                                                    kernel_size=self.ksizes[0], 
                                                    abs_bnorm=False, bias=self.bias, 
                                                    stack_chan=self.stack_chans[0], 
                                                    stack_ksize=self.stack_ksizes[0],
                                                    drop_p=self.drop_p, 
                                                    padding=self.paddings[0]))
        shape = update_shape(shape, self.ksizes[0], padding=self.paddings[0])
        self.shapes.append(tuple(shape))
        # BatchNorm
        if self.bnorm_d == 1:
            modules.append(Flatten())
            modules.append(AbsBatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, 
                                                            momentum=self.bn_moment))
            modules.append(Reshape((-1,self.chans[0],shape[0], shape[1])))
        else:
            modules.append(AbsBatchNorm2d(self.chans[0], eps=1e-3, momentum=self.bn_moment))
        # Noise and ReLU
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())

        ##### Second Layer
        # Convolution
        if one2one:
            modules.append(OneToOneLinearStackedConv2d(self.chans[0],self.chans[1],
                                                        kernel_size=self.ksizes[1], 
                                                        padding=self.paddings[1],
                                                        bias=self.bias))
        else:
            modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],
                                                    kernel_size=self.ksizes[1], 
                                                    abs_bnorm=False, bias=self.bias, 
                                                    stack_chan=self.stack_chans[1], 
                                                    stack_ksize=self.stack_ksizes[1],
                                                    padding=self.paddings[1],
                                                    drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1], padding=self.paddings[1])
        self.shapes.append(tuple(shape))
        # BatchNorm
        if self.bnorm_d == 1:
            modules.append(Flatten())
            modules.append(AbsBatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, 
                                                        momentum=self.bn_moment))
        else:
            modules.append(AbsBatchNorm2d(self.chans[1], eps=1e-3, momentum=self.bn_moment))
            modules.append(Flatten())
        # Noise and ReLU
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())

        ##### Final Layer
        if self.convgc:
            modules.append(Reshape((-1, self.chans[1], shape[0], shape[1])))
            modules.append(nn.Conv2d(self.chans[1],self.n_units,kernel_size=self.ksizes[2], 
                                                                    bias=self.linear_bias))
            shape = update_shape(shape, self.ksizes[2])
            self.shapes.append(tuple(shape))
            modules.append(GrabUnits(self.centers, self.ksizes, self.img_shape, self.n_units))
            modules.append(AbsBatchNorm1d(self.n_units, momentum=self.bn_moment))
        else:
            modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, 
                                                                bias=self.linear_bias))
            modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        if self.final_bias:
            modules.append(Add(0,trainable=True))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    
    def tiled_forward(self,x):
        """
        Allows for the fully convolutional functionality
        """
        if not self.convgc:
            return self.forward(x)
        fx = self.sequential[:-3](x) # Remove GrabUnits layer
        bnorm = self.sequential[-2]
        # Perform 2d batchnorm using 1d parameters collected from training
        fx = torch.nn.functional.batch_norm(fx, bnorm.running_mean.data, bnorm.running_var.data,
                                                    weight=bnorm.scale.abs(), bias=bnorm.shift, 
                                                    eps=bnorm.eps, momentum=bnorm.momentum, 
                                                    training=self.training)
        fx = self.sequential[-1](fx)
        if not self.training and self.infr_exp:
            return torch.exp(fx)
        return fx

class SimpleClassifier(nn.Module):
    def __init__(self, n_classes=7, h_size=256, c_size=256, **kwargs):
        super().__init__()
        self.c_size = c_size
        self.h_size = h_size
        self.n_classes = n_classes
        modules = []
        modules.append(nn.Linear(self.h_size, self.c_size))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.c_size, self.n_classes))
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)

class MotherModel(nn.Module):
    """
    Wraps the LinearStackedBNCNN and creates a video classifcation LSTM
    """
    def __init__(self, **kwargs):
        super().__init__()
        # TODO: Add switch for training dr model
        model = globals()[kwargs['dr_type']](**kwargs)
        if kwargs['saved_model_path'] is not None:
            data = torch.load(kwargs['saved_model_path'], map_device="cpu")
            model.load(data['model_state_dict'])
        self.h_size = kwargs['h_size']
        self.rnn = globals()[kwargs['rnn_type']](kwargs['n_units'], self.h_size)
        self.classifier = globals()[kwargs['classifier_type']](kwargs['c_size'])

    def init_h(self,batch_size):
        if isinstance(self.rnn, nn.LSTMCell):
            h = torch.zeros(batch_size, self.h_size)
            c = torch.zeros(batch_size, self.h_size)
            if next(self.parameters()).is_cuda:
                h = h.to(DEVICE)
                c = c.to(DEVICE)
            return (h,c)
        else:
            raise NotImplementedError

    def forward(self, x, h):
        fx = self.dr(x)
        h = self.rnn(fx,h)
        return h

    def classify(self, h):
        return self.classifier(h[0])

def get_deep_retina(save_name=None, **kwargs):
    """
    Handles loading previous models if necessary
    """
    return model

