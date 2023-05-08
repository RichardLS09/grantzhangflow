"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return (MakeTensorTuple()(*in_grad),)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        out = (a + b)
        # assert out.dtype == numpy.float32
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        def _adajust(out,aim):
          sum_axis = []
          for index,(i,j) in enumerate(zip(out.shape,aim.shape)):
            if i!=j:
              assert i>j
              sum_axis.append(index)
          if len(sum_axis) == 0:
            return out
          return reshape(summation(out,tuple(sum_axis)),aim.shape)

        a,b = (_adajust(out_grad,lhs),_adajust(out_grad,rhs))
        return a,b


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        out = a + self.scalar
        # assert out.dtype == numpy.float32
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        o1 = out_grad * rhs
        o2 = out_grad * lhs

        def _adajust(out,aim):
          sum_axis = []
          for index,(i,j) in enumerate(zip(out.shape,aim.shape)):
            if i!=j:
              assert i>j
              sum_axis.append(index)
          if len(sum_axis) == 0:
            return out
          return reshape(summation(out,tuple(sum_axis)),aim.shape)

        return (_adajust(o1,lhs),_adajust(o2,rhs))


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        # return a * numpy.float32(self.scalar)
        out = a * self.scalar
        # assert out.dtype == numpy.float32
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        out = a ** (self.scalar)
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, *others = node.inputs
        return (
          mul_scalar(power_scalar(lhs, self.scalar-1),self.scalar) *
          out_grad
          ,
        )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        ### BEGIN YOUR SOLUTION
        out = a/b
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        o1 = out_grad/rhs
        o2 = out_grad * (-lhs/power_scalar(rhs,2))

        def _adajust(out,aim):
          sum_axis = []
          for index,(i,j) in enumerate(zip(out.shape,aim.shape)):
            if i!=j:
              assert i>j
              sum_axis.append(index)
          if len(sum_axis) == 0:
            return out
          return reshape(summation(out,tuple(sum_axis)),aim.shape)

        return (_adajust(o1,lhs),_adajust(o2,rhs))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # out = (a/self.scalar).astype(a.dtype)
        out = a/self.scalar
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad/self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
          nn = list(range(len(a.shape)))
          nn[-2:] = nn[-2:][::-1]
          out = array_api.transpose(a,nn)
          out = out.compact()
          return out
        if len(a.shape) == len(self.axes):
          out = array_api.transpose(a,self.axes)
          out = out.compact()
          return out
        
        # don't match
        assert len(self.axes) == 2
        # o1,o2 = self.axes
        # return array_api.swapaxes(a,o1,o2)
        new_axes = list(range(len(a.shape)))
        for i_index,i in enumerate(new_axes):
            if i == self.axes[0]:
                new_axes[i_index] = self.axes[1]
            elif i == self.axes[1]:
                new_axes[i_index] = self.axes[0]
        out = array_api.transpose(a,new_axes)
        # assert out.dtype == numpy.float32
        out = out.compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_axes = self.axes
        # lrs,*tmp = node.inputs
        if new_axes is not None and len(new_axes) == len(out_grad.shape):
          sd_axes = list(range(len(new_axes)))
          new_axes = [new_axes.index(i) for i in sd_axes]
        out = transpose(out_grad,new_axes)
        return (out,)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a = a.compact()
        out = array_api.reshape(a,self.shape)
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs , *others = node.inputs
        return (reshape(out_grad,lhs.shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, *others = node.inputs
        lhs_shape = list(lhs.shape)
        # out_grad_shape = out_grad.shape
        aim_shape = list(self.shape)
        # assert aim_shape[-len(lhs_shape):] == lhs_shape[:]
        assert len(aim_shape[-len(lhs_shape):]) == len(lhs_shape[:])
        n_shape = list(range(len(aim_shape)-len(lhs_shape)))
        for index,(i,j) in enumerate(zip(aim_shape[-len(lhs_shape):],lhs_shape[:])):
          if i == j:
            continue
          assert j == 1
          n_shape.append(index)
          

        # n_shape = []
        # for i in range(len(aim_shape)-len(lhs_shape)):
        #     n_shape.append(i)
        if len(n_shape) == 0:
            out = out_grad
        else:
            t = out_grad
            for i in n_shape:
              t = summation(t,(i,))
              t_shape = aim_shape[:]
              t_shape[i] = 1
              t = reshape(t,t_shape)
            out = t
            if list(out.shape)!=lhs_shape:
              out = reshape(out,lhs_shape)
        return (out,)
        # lhs,*others = node.inputs
        # n_shape = []
        # if len(lhs.shape) == len(self.shape):
        #   for i_index,(i,j) in enumerate(zip(lhs.shape,self.shape)):
        #     if i!=j:
        #       n_shape.append(i_index)
        # elif len(lhs.shape) < len(self.shape):
        #   i_index = -1
        #   for i_index,i in enumerate(lhs.shape):
        #     if i!= self.shape[i_index]:
        #       n_shape.append(i_index)
        #   for i in range(i_index+1,len(self.shape)):
        #     n_shape.append(i)
        # if len(n_shape) == 0:
        #   n_shape = None
        # out = summation(out_grad,tuple(n_shape))
        # return (reshape(out,lhs.shape),)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out = a.sum(axis=self.axes,keepdims=False)
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,*others = node.inputs
        new_shape = [1] * len(lhs.shape)
        
        if self.axes is not None:
          for i_index,i in enumerate(lhs.shape):
            if i_index not in self.axes:
              new_shape[i_index] = i
        out = broadcast_to(
          reshape(out_grad,new_shape),
          lhs.shape)

        return (out,)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    if axes is not None and isinstance(axes,int):
      axes = (axes,)
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        out = array_api.matmul(a,b)
        # print(a.dtype,b.dtype)
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs

        # return (
        #   matmul(out_grad,transpose(rhs)),
        #   matmul(transpose(lhs),out_grad),
        #   )
        def _adajust(o1,aim):
          if len(o1.shape) == len(aim.shape):
            return o1
          elif len(o1.shape) < len(aim.shape):
            return broadcast_to(o1,aim.shape)
          else:
            new_axis = tuple(range(len(o1.shape)-len(aim.shape)))
            return summation(o1,new_axis)
        return (
          _adajust(matmul(out_grad,transpose(rhs)),lhs),
          _adajust(matmul(transpose(lhs),out_grad),rhs),
        )
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out = -a
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-1 * out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out = array_api.log(a)
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,*others = node.inputs
        return (out_grad/lhs,)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out = array_api.exp(a)
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,*others = node.inputs
        return (exp(lhs)*out_grad,)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # out = array_api.where(a>0.0,a,0.0)
        # assert out.dtype == numpy.float32
        # return out.astype(a.dtype)
        out = array_api.maximum(a,0.0)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dd = node.inputs[0].realize_cached_data()
        out = Tensor.make_const(
          # array_api.where(dd>0.0,1.0,0.0).astype(dd.dtype),
          # array_api.maximum(dd,0.0)
          dd >= 0.0
        )
        return (out_grad * out,)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(self.axes,keepdims=False)
        shape = Z.shape

        if self.axes is None:
            b_shape = [1]*len(shape)
        else:
            b_shape = []
            for index,i in enumerate(shape):
                if index in self.axes:
                    b_shape.append(1)
                else:
                    b_shape.append(i)
        b_m_z = array_api.reshape(max_z,b_shape)
        b_m_z = array_api.broadcast_to(b_m_z,list(Z.shape))

        s_x = array_api.log((array_api.exp(Z-b_m_z)).sum(self.axes,keepdims=False))
        s_out = s_x + max_z
        out = s_out
        # assert out.dtype == numpy.float32
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,*rhs = node.inputs
        inter_grad = logsumexp(lhs,self.axes)

        shape = lhs.shape
        if self.axes is None:
            b_shape = [1]*len(shape)
        else:
            b_shape = []
            for index,i in enumerate(shape):
                if index in self.axes:
                    b_shape.append(1)
                else:
                    b_shape.append(i)
        inter_grad_re = reshape(inter_grad, b_shape)
        inter_grad_log = lhs - broadcast_to(inter_grad_re, lhs.shape)
        inter_grad_exp = exp(inter_grad_log)
        out_grad_re = reshape(out_grad, b_shape)
        out_grad_re = broadcast_to(out_grad_re, inter_grad_exp.shape)
        grad = multiply(out_grad_re,inter_grad_exp)
        return (grad,)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    if axes is not None and isinstance(axes,int):
      axes = (axes,)
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out = 2.0 * ((1 + array_api.exp(-2 * a)) ** -1) - 1
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lrs,*others = node.inputs
        out = out_grad * (-(tanh(lrs) ** 2) + 1)
        # print(out_grad * (1 - (tanh(lrs) ** 2)))
        return (out,)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = array_api.stack(args, self.axis)
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad_shape = list(out_grad.shape)
        new_shape = []
        for index,i in enumerate(out_grad_shape):
           if index != self.axis:
            new_shape.append(i)
        out = split(out_grad,self.axis)
        out = [reshape(i,new_shape) for i in out]
        return (make_tuple(*out),)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out = array_api.split(A, self.axis)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = stack(out_grad, self.axis)

        return (reshape(out,node.inputs[0].shape),)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad,self.axes),)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        new_shape = []
        new_slices = []
        for index,i in enumerate(old_shape):
          this_shape = i if index not in self.axes else i + self.dilation * i
          new_shape.append(
            this_shape
          )
          new_slices.append(
            slice(0,this_shape,1 if index not in self.axes else self.dilation + 1)
          )

        out = array_api.full(new_shape,0.0,a.dtype,a.device)
        out[tuple(new_slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = undilate(out_grad, self.axes, self.dilation)
        return (out,)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        new_shape = []
        new_slices = []
        for index,i in enumerate(old_shape):
          this_shape = i if index not in self.axes else i // (self.dilation + 1)
          new_shape.append(
            this_shape
          )
          new_slices.append(
            slice(0,i,1 if index not in self.axes else self.dilation + 1)
          )

        out = array_api.empty(new_shape,a.dtype,a.device)
        out = a[tuple(new_slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = dilate(out_grad, self.axes, self.dilation)
        return (out,)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: NHWC
        # B: k,k,inp,out

        if self.padding != 0:
          pad_A = A.pad(
            (
              (0,0),
              (self.padding,self.padding),
              (self.padding,self.padding),
              (0,0)
            )
          )
        else:
          pad_A = A
        
        N,H,W,C_in = list(pad_A.shape)
        K,_,_,C_out = list(B.shape)
        Ns,Hs,Ws,Es = list(pad_A.strides)

        new_shape = (N,
          (H-K)//self.stride+1,
          (W-K)//self.stride+1,
          K,K,C_in
        )
        new_strides = (Ns,
          self.stride * Hs,self.stride * Ws,
          Hs,Ws, 
          Es
        )
        new_A = pad_A.as_strided(
          new_shape,
          new_strides
        ).compact()

        inner_dim = K * K * C_in
        new_A = new_A.reshape(
          (
            new_shape[0] * new_shape[1] * new_shape[2]
            ,inner_dim
          )
        )

        new_B = B.reshape(
          (
            inner_dim, C_out
          )
        )

        out = new_A @ new_B

        out = out.reshape(
          (N,new_shape[1],new_shape[2],C_out)
        )
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs

        N,H,W,C_in = list(lhs.shape)
        K,_,_,C_out = list(rhs.shape)
        _,NH,NW,_ = list(out_grad.shape)

        use_out_grad = out_grad
        if self.stride > 1:
          # assert False
          num_stride = self.stride - 1
          use_out_grad = dilate(
            use_out_grad,(1,2),
            num_stride
          )
        
        num_stride = 1
        num_padding = K - 1 - self.padding
        new_w = rhs

        new_w = transpose(new_w,(0,1,3,2))
        new_w = flip(new_w, (0,1))

        out1 = conv(
          use_out_grad,
          new_w,
          stride=num_stride, 
          padding=num_padding
        )


        num_stride = 1
        num_padding = self.padding
        acc_grad = transpose(out_grad,
          (1,2,0,3)
        )
        new_x = transpose(
            lhs, (3,1,2,0)
        )
        if self.stride > 1:
          acc_grad = dilate(
            acc_grad,(0,1),self.stride - 1
          )

        out2 = conv(
          new_x,
          acc_grad,
          stride=num_stride, 
          padding=num_padding
        )
        out2 = transpose(
          out2, (1,2,0,3)
        )
        
        return (out1,out2)

        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



