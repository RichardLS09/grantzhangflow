"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(in_features,
          out_features,device=device, dtype=dtype
        )
        self.weight = Parameter(self.weight.numpy(),device=device, dtype=dtype)
        self._use_bias = bias
        if bias == True:
          self.bias = init.kaiming_uniform(
            out_features,1,device=device, dtype=dtype
          ).transpose()
          self.bias = Parameter(self.bias.numpy(),device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        tmp = X@self.weight
        if self._use_bias:
          if len(self.bias.shape)!=len(tmp.shape):
            assert len(tmp.shape) > len(self.bias.shape)
            assert False
            new_shape = [1]*(len(tmp.shape) - len(self.bias.shape)) + list(self.bias.shape)
            tmp += ops.reshape(self.bias, new_shape)
          else:
            tmp += ops.broadcast_to(self.bias,tmp.shape)
        return tmp
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        b_shape = [shape[0],]
        if len(shape)>1:
          n = 1
          for i in range(1,len(shape)):
            n *= shape[i]
          b_shape.append(n)
        return ops.reshape(X,b_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e_x = ops.exp(-x)
        e_x += 1
        out = ops.power_scalar(e_x, -1)
        return out
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for i in self.modules:
          out = i(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        shape = logits.shape
        assert len(shape) == 2
        tmp = ops.logsumexp(logits,axes=(1,))
        tmp_r = ops.reshape(tmp,(shape[0],1))
        tmp_r = ops.broadcast_to(tmp_r, logits.shape)
        tmp_log = tmp_r - logits
        labels = init.one_hot(shape[1], y, dtype=logits.dtype, device=y.device)
        s_sum = ops.summation(tmp_log * labels) / shape[0]
        return s_sum
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(
            dim, device=device, dtype=dtype,
            requires_grad=True
        )
        self.bias = init.zeros(
            dim, device=device, dtype=dtype,
            requires_grad=True
        )
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype,
            requires_grad=False
        )
        self.running_var = init.ones(
            dim, device=device, dtype=dtype,
            requires_grad=False
        )
        self.weight = Parameter(self.weight.numpy(),device=device, dtype=dtype)
        self.bias = Parameter(self.bias.numpy(),device=device, dtype=dtype)
        self.running_mean = self.running_mean
        self.running_var = self.running_var
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        def mean(res, shape, broad_shape):
            s_sum = ops.summation(res, (0,))
            s_sum_r = ops.reshape(s_sum, broad_shape)
            return s_sum_r / shape[0]

        def get_mean_var(res, shape, broad_shape):
            res_mean = mean(res, shape, broad_shape)
            # res_var = mean((res*res), shape, broad_shape) - res_mean ** 2
            # res_var = mean((res-res_mean.detach()) ** 2,shape, broad_shape)
            b_res_mean = ops.broadcast_to(res_mean,res.shape)
            res_var = mean((res-b_res_mean) ** 2,shape, broad_shape)
            n = shape[0]
            # res_var = res_var * n / (n-1)
            return res_mean, res_var

        shape = x.shape
        n_dim = len(shape)
        broad_shape = [1] * n_dim
        for i in range(1,n_dim):
            broad_shape[i] = shape[i]

        if self.training:
            this_mean, this_var = get_mean_var(x, shape, broad_shape)
            self.running_mean.data = (
                self.momentum * ops.reshape(this_mean.data,self.running_mean.shape).data +
                (1-self.momentum) * self.running_mean.data
            )
            self.running_var.data = (
                self.momentum * ops.reshape(this_var.data,self.running_var.shape).data +
                (1-self.momentum) * self.running_var.data
            )
            use_mean,use_var = this_mean, this_var
        else:
          use_mean = ops.reshape(self.running_mean,(1,self.dim))
          use_var = ops.reshape(self.running_var,(1,self.dim))
        
        
        use_mean = ops.broadcast_to(use_mean, shape)
        
        inv = ops.power_scalar(use_var + self.eps, 0.5)
        inv = ops.reshape(self.weight, (1, self.dim)) / inv

        inv = ops.broadcast_to(inv, shape)

        b_bias = ops.reshape(self.bias, (1, self.dim))
        b_bias = ops.broadcast_to(b_bias, shape)

        return x * inv + (b_bias - use_mean * inv)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(
            dim, device=device, dtype=dtype,
            requires_grad=True
        )
        self.bias = init.zeros(
            dim, device=device, dtype=dtype,
            requires_grad=True
        )
        self.weight = Parameter(self.weight.numpy(),device=device, dtype=dtype)
        self.bias = Parameter(self.bias.numpy(),device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        def mean(res, shape, broad_shape):
            n_dim = len(shape)
            s_sum = ops.summation(res, (n_dim-1,))
            s_sum_r = ops.reshape(s_sum, broad_shape)
            return s_sum_r / shape[n_dim-1]

        shape = x.shape
        n_dim = len(shape)
        broad_shape = [1] * n_dim
        for i in range(n_dim-1):
            broad_shape[i] = shape[i]

        x_mean = mean(x, shape, broad_shape)
        # x_var = mean((x*x), shape, broad_shape) - x_mean ** 2
        b_x_mean = ops.broadcast_to(x_mean, shape)
        x_var = mean((x-b_x_mean)**2,shape, broad_shape)
        x_var = ops.broadcast_to(x_var, shape)
        m_x = (x - b_x_mean) / ops.power_scalar(
            x_var + self.eps, 1/2
        )
        b_weight = ops.reshape(self.weight, (1, self.dim))
        b_bias = ops.reshape(self.bias, (1, self.dim))

        b_weight = ops.broadcast_to(b_weight, shape)
        b_bias = ops.broadcast_to(b_bias, shape)
        return  b_weight* m_x + b_bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(
                *x.shape, p = 1 - self.p, device=x.device,
                dtype=x.dtype, requires_grad=False
            )
            return mask * x  / (1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        w = init.kaiming_uniform(
          kernel_size*kernel_size*in_channels,
          kernel_size*kernel_size*out_channels,
          shape=[kernel_size,kernel_size,in_channels,out_channels],
          dtype=dtype,device=device
        )
        self.weight = Parameter(w.numpy(),device=device, dtype=dtype)
        self._use_bias = bias
        if self._use_bias:
          b = init.rand(
            out_channels,
            low=-1.0/(in_channels * kernel_size**2)**0.5,
            high=1.0/(in_channels * kernel_size**2)**0.5,
            dtype=dtype,device=device
          )
          self.bias = Parameter(b.numpy(),device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N,C,H,W = x.shape
        
        r_x = ops.transpose(x, [0,2,3,1])

        out = ops.conv(r_x,self.weight,stride=self.stride,
          padding=self.kernel_size//2)

        if self._use_bias:
          b_bias = ops.reshape(self.bias,[1,1,1,self.out_channels])
          b_bias = ops.broadcast_to(b_bias, out.shape)

          out += b_bias
        
        out = ops.transpose(out, [0,3,1,2])
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size
        self.hidden_size=hidden_size
        self._use_bias = bias
        self.nonlinearity = nonlinearity

        assert self.nonlinearity in ("tanh","relu")

        k = 1/hidden_size
        import math
        kk = math.sqrt(k)

        def get_p(*shape,):
          tmp = init.rand(
            *shape,low=-kk,high=kk,
            device=device,dtype=dtype
          )
          return Parameter(tmp.numpy(),device=device,dtype=dtype)
        self.W_ih = get_p(input_size, hidden_size)
        self.W_hh = get_p(hidden_size, hidden_size)
        self.bias_ih = None
        self.bias_hh = None
        if self._use_bias:
          self.bias_ih = get_p(hidden_size)
          self.bias_hh = get_p(hidden_size)
        
        self.act = ops.ReLU() if self.nonlinearity == "relu" else ops.Tanh()


        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        x_shape = list(X.shape)
        if h is None:
          h = init.zeros(
            x_shape[0],self.hidden_size,
            device=X.device,
            dtype=X.dtype,
          )
        
        def bias_add(inp,w,b):
          out = inp@w
          if b is not None:
            b = ops.reshape(b,[1,self.hidden_size])
            b = ops.broadcast_to(b, out.shape)
            out += b
          return out
        out = bias_add(X,self.W_ih,self.bias_ih) + \
            bias_add(h,self.W_hh,self.bias_hh)
        return self.act(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn_cells=[]
        for i in range(self.num_layers):
          inp = input_size if i == 0 else hidden_size
          self.rnn_cells.append(
            RNNCell(inp, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, 
            dtype=dtype)
          )

        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        x_shape = X.shape
        seq_len, bs, input_size = x_shape
        if h0 is None:
          h0 = init.zeros(
            self.num_layers,bs,self.hidden_size,
            dtype=X.dtype,device=X.device
          )
        h_split = ops.split(h0,0)

        def one_step(inp,cell,h):
          seq_len, bs, input_size = inp.shape

          out = []
          inp_split = ops.split(inp,0)
          for i in range(seq_len):
            t_inp= inp_split[i]
            t_inp = ops.reshape(t_inp,[bs, input_size])
            h = cell(t_inp,h)
            out.append(h)
          
          return ops.stack(out,0),h
        
        this_out = X
        h_out = []
        for i in range(self.num_layers):
          this_out,this_h = one_step(
            this_out,self.rnn_cells[i],
            ops.reshape(h_split[i],list(h0.shape)[1:])
          )
          h_out.append(this_h)
        
        return this_out,ops.stack(h_out,0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size
        self.hidden_size=hidden_size
        self._use_bias=bias

        k = 1/hidden_size
        import math
        kk = math.sqrt(k)
        def get_p(*shape,):
          tmp = init.rand(
            *shape,low=-kk,high=kk,
            device=device,dtype=dtype
          )
          return Parameter(tmp.numpy(),device=device,dtype=dtype)
        
        self.W_ih = get_p(input_size, 4*hidden_size)
        self.W_hh = get_p(hidden_size, 4*hidden_size)
        self.bias_ih = None
        self.bias_hh = None
        if self._use_bias:
          self.bias_ih = get_p(4*hidden_size)
          self.bias_hh = get_p(4*hidden_size)
        
        self.act_i = Sigmoid()
        self.act_f = Sigmoid()
        self.act_g = ops.Tanh()
        self.act_o = Sigmoid()

        self.act_c = ops.Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        x_shape = X.shape
        bs = x_shape[0]
        if h is None:
          h0 = init.zeros(
            bs,self.hidden_size,
            device=X.device,
            dtype=X.dtype
          )
          c0 = init.zeros(
            bs,self.hidden_size,
            device=X.device,
            dtype=X.dtype
          )
          h = [h0,c0]
        h0,c0 = h

        def split(inp,is_bias=False):
          if inp is None:
            return [None]*4
          if is_bias:
            inp = ops.reshape(inp,[1,inp.shape[0]])
          inp_list = ops.split(inp,1)
          new_shape = [inp.shape[0],self.hidden_size]
          w_list = [
            ops.stack([inp_list[j] 
            for j in range(i*self.hidden_size,
            i*self.hidden_size+self.hidden_size)],1)
            for i in range(4)
          ]
          w_list = [ops.reshape(i,new_shape) for i in w_list]
          return w_list
        w_ih = split(self.W_ih)
        w_hh = split(self.W_hh)
        b_ih = split(self.bias_ih,True)
        b_hh = split(self.bias_hh,True)

        def bias_add(inp):
          ih,hh,b_i,b_h = inp
          tmp = X@ih
          if b_i is not None:
            tmp += ops.broadcast_to(b_i,tmp.shape)
          tmp += h0@hh
          if b_h is not None:
            tmp += ops.broadcast_to(b_h,tmp.shape)
          return tmp
        i,f,g,o = [bias_add(m) for m in zip(w_ih,w_hh,b_ih,b_hh)]
        i = self.act_i(i)
        f = self.act_f(f)
        g = self.act_g(g)
        o = self.act_o(o)

        c = f * c0 + i * g
        h = o * self.act_c(c)
        return (h,c)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm_cells=[]
        for i in range(self.num_layers):
          inp = input_size if i == 0 else hidden_size
          self.lstm_cells.append(
            LSTMCell(inp, hidden_size, bias=bias, device=device, 
            dtype=dtype)
          )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_shape = X.shape
        seq_len, bs, input_size = X_shape
        if h is None:
          h0 = init.zeros(
            self.num_layers,bs,self.hidden_size,
            dtype=X.dtype,device=X.device
          )
          c0 = init.zeros(
            self.num_layers,bs,self.hidden_size,
            dtype=X.dtype,device=X.device
          )
          h = [h0,c0]
        
        h0,c0 = h
        h_split = ops.split(h0,0)
        c_split = ops.split(c0,0)

        def one_step(inp,cell,h):
          seq_len, bs, input_size = inp.shape

          out = []
          inp_split = ops.split(inp,0)
          for i in range(seq_len):
            t_inp= inp_split[i]
            t_inp = ops.reshape(t_inp,[bs, input_size])
            h = cell(t_inp,h)
            out.append(h[0])
          
          return ops.stack(out,0),h
        
        this_out = X
        h_out = []
        for i in range(self.num_layers):
          this_out,this_h = one_step(
            this_out,self.lstm_cells[i],
            [ops.reshape(h_split[i],list(h0.shape)[1:]),
            ops.reshape(c_split[i],list(c0.shape)[1:]),
            ]
          )
          h_out.append(this_h)
        h_out = [
          ops.stack(i,0)
          for i in zip(*h_out)
        ]
        return this_out,tuple(h_out)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        def get_p(*shape,):
          tmp = init.randn(
            *shape,
            device=device,dtype=dtype
          )
          return Parameter(tmp.numpy(),device=device,dtype=dtype)
        self.weight = get_p(
          num_embeddings, embedding_dim
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_shape = x.shape
        seq_len, bs = x_shape

        f_x = ops.reshape(x,[seq_len*bs,])

        o_x = init.one_hot(
          self.num_embeddings, f_x, device=x.device, 
          dtype=x.dtype, requires_grad=True
        )
        out = o_x@self.weight
        out = ops.reshape(out,[seq_len, bs, self.embedding_dim])
        return out
        ### END YOUR SOLUTION