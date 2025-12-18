import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from torch.nn import Module 
from torch.nn.parameter import Parameter 
import torch.nn as nn
from quaternion_ops import *  



class InvalidInput(RuntimeError): 
  def __init__(self, error_message):
    super().__init__(error_message)

class QuaternionConv(nn.Module):
  ALLOWED_DIMENSIONS = (2, 3)  
  def __init__(self, in_channels, out_channels, kernel_size, stride, dimension=2, padding=0, dilation=1, groups=1, bias=True): 
    super(QuaternionConv, self).__init__()  
    self.in_channels = np.floor_divide(in_channels, 4) 
    self.out_channels = np.floor_divide(out_channels, 4)  
    self.groups = groups  
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.kernel_size = self.get_kernel_shape(kernel_size, dimension)  
    self.weight_shape = self.get_weight_shape(self.in_channels, self.out_channels, self.kernel_size) 
    self._weights = self.weight_tensors(self.weight_shape, kernel_size) 
    self.r_weight, self.k_weight, self.i_weight, self.j_weight = self._weights 
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))  
      nn.init.constant_(self.bias, 0) 

  def forward(self, x):
    """Apply forward pass of input through quaternion convolution layer."""
    cat_kernels_4_r = torch.cat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=1) 
    cat_kernels_4_i = torch.cat([self.i_weight,  self.r_weight, -self.k_weight, self.j_weight], dim=1)  
    cat_kernels_4_j = torch.cat([self.j_weight,  self.k_weight, self.r_weight, -self.i_weight], dim=1)
    cat_kernels_4_k = torch.cat([self.k_weight,  -self.j_weight, self.i_weight, self.r_weight], dim=1)
    cat_kernels_4_quaternion   = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0) 

    if x.dim() == 3:
        convfunc = F.conv1d
    elif x.dim() == 4:
        convfunc = F.conv2d
    elif x.dim() == 5:
        convfunc = F.conv3d
    else:
        raise InvalidInput("Given input channels do not match allowed dimensions")
    return convfunc(x, cat_kernels_4_quaternion, self.bias, self.stride, self.padding, self.dilation, self.groups)

  @staticmethod
  def weight_tensors(weight_shape, kernel_size):
    modulus = nn.Parameter(torch.Tensor(*weight_shape))  
    modulus = nn.init.xavier_uniform_(modulus, gain=1.0) 
    i_weight = 2.0 * torch.rand(*weight_shape) - 1.0
    j_weight = 2.0 * torch.rand(*weight_shape) - 1.0
    k_weight = 2.0 * torch.rand(*weight_shape) - 1.0
    sum_imaginary_parts = i_weight.abs() + j_weight.abs() + k_weight.abs()
    i_weight = torch.div(i_weight, sum_imaginary_parts)
    j_weight = torch.div(j_weight, sum_imaginary_parts)
    k_weight = torch.div(k_weight, sum_imaginary_parts)
    phase = torch.rand(*weight_shape) * (2 * torch.tensor([np.pi])) - torch.tensor([np.pi])
    r_weight = modulus * np.cos(phase)
    i_weight = modulus * i_weight * np.sin(phase)
    j_weight = modulus * j_weight * np.sin(phase)
    k_weight = modulus * k_weight * np.sin(phase)
    return nn.Parameter(r_weight), nn.Parameter(i_weight), nn.Parameter(j_weight), nn.Parameter(k_weight)

  @staticmethod 
  def get_weight_shape(in_channels, out_channels, kernel_size): 
    return (out_channels, in_channels) + kernel_size

  @staticmethod
  def get_kernel_shape(kernel_size, dimension): 
    if dimension not in QuaternionConv.ALLOWED_DIMENSIONS:
      raise InvalidKernelShape('Given dimensions are not allowed.')
    if isinstance(kernel_size, int):
      return (kernel_size, ) * dimension

    if isinstance(kernel_size, tuple):
      if len(kernel_size) != dimension:
        raise InvalidKernelShape('Given kernel shape does not match dimension.')

      return kernel_size

  def __repr__(self):
      return self.__class__.__name__ + '(' \
          + 'in_channels='      + str(self.in_channels) \
          + ', out_channels='   + str(self.out_channels) \
          + ', kernel_size='    + str(self.kernel_size) \
          + ', stride='         + str(self.stride) + ')'


class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):  
        super(QuaternionBatchNorm2d, self).__init__()   
        self.num_features = num_features // 4   
        self.gamma_init = gamma_init   
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))  
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)  
        self.training = training
        self.eps = torch.tensor(1e-5)  

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)  
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        delta_r, delta_i, delta_j, delta_k = r - torch.mean(r), i - torch.mean(i), j - torch.mean(j), k - torch.mean(k) 
        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2))
        denominator = torch.sqrt(quat_variance + self.eps) 
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1) 
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1) 
        return new_input
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'

class QuaternionReLU(Module):
    def __init__(self):
        super(QuaternionReLU, self).__init__()

    def forward(self, input):        
        activated_components = [torch.relu(input[..., i:i+1]) for i in range(input.size(-1))]
        output = torch.cat(activated_components, dim=-1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class QuaternionSigmoid(Module):
    def __init__(self):
        super(QuaternionSigmoid, self).__init__()

    def forward(self, input):
        if input.size(-1) % 4 != 0:
            raise ValueError("The last dimension of the input must be divisible by 4 to represent quaternions.")
        r, i, j, k = torch.chunk(input, chunks=4, dim=-1)
        r = torch.sigmoid(r)
        i = torch.sigmoid(i)
        j = torch.sigmoid(j)
        k = torch.sigmoid(k)
        output = torch.cat([r, i, j, k], dim=-1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class QuaternionMaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(QuaternionMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, input):
        if input.size(1) % 4 != 0:
            raise ValueError("The channel dimension of the input must be divisible by 4 to represent quaternions.")
        r, i, j, k = torch.chunk(input, chunks=4, dim=1)
        r = torch.nn.functional.max_pool2d(r, self.kernel_size, self.stride, self.padding)
        i = torch.nn.functional.max_pool2d(i, self.kernel_size, self.stride, self.padding)
        j = torch.nn.functional.max_pool2d(j, self.kernel_size, self.stride, self.padding)
        k = torch.nn.functional.max_pool2d(k, self.kernel_size, self.stride, self.padding)
        output = torch.cat([r, i, j, k], dim=1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) + ')'

class QuaternionAvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(QuaternionAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, input):
        if input.size(1) % 4 != 0:
            raise ValueError("The channel dimension of the input must be divisible by 4 to represent quaternions.")

        r, i, j, k = torch.chunk(input, chunks=4, dim=1)
        r = torch.nn.functional.avg_pool2d(r, self.kernel_size, self.stride, self.padding)
        i = torch.nn.functional.avg_pool2d(i, self.kernel_size, self.stride, self.padding)
        j = torch.nn.functional.avg_pool2d(j, self.kernel_size, self.stride, self.padding)
        k = torch.nn.functional.avg_pool2d(k, self.kernel_size, self.stride, self.padding)
        output = torch.cat([r, i, j, k], dim=1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) + ')'

class QuaternionLinear(Module):
    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()   
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()
    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)    
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
            output = output.view(T, N, output.size(1))  
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) + ')'

class QuaternionLinearReal(Module):
    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinearReal, self).__init__()    
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        if input.dim() == 3:
            T, N, C = input.size()  
            input = input.view(T * N, C)    
            output = QuaternionLinearReal.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
            magnitudes = torch.sqrt(output[:, 0]**2 + output[:, 1]**2 + output[:, 2]**2 + output[:, 3]**2)  
            output = output.view(T, N, output.size(1))  
        elif input.dim() == 2:
            output = QuaternionLinearReal.apply(input,self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
            magnitudes = torch.sqrt(output[:, 0]**2 + output[:, 1]**2 + output[:, 2]**2 + output[:, 3]**2)  
        else:
            raise NotImplementedError
        
        

        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) + ')'
               
