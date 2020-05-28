# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================

import math
import numpy as np
from functools import wraps

from singa import utils
from .tensor import Tensor
from . import singa_wrap as singa


class LayerMeta(type):

    def init_wrapper(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if len(args) == 0:
                return

            if isinstance(args[0], list):
                assert len(args) > 0 and isinstance(args[0][0], Tensor), (
                    'initialize function expects PlaceHolders or Tensors')
                dev = args[0][0].device
            else:
                assert len(args) > 0 and isinstance(args[0], Tensor), (
                    'initialize function expects PlaceHolders or Tensors')
                dev = args[0].device

            prev_state = dev.graph_enabled()
            dev.EnableGraph(False)
            func(self, *args, **kwargs)

            # sanitize the dev of params/states init-ed
            for s in self.state_names:
                if isinstance(args[0], list):
                    self.device_check(args[0][0], self.__dict__[s])
                else:
                    self.device_check(args[0], self.__dict__[s])

            self._initialzied = True
            dev.EnableGraph(prev_state)

        return wrapper

    def __new__(cls, name, bases, attr):
        if 'initialize' in attr:
            attr['initialize'] = LayerMeta.init_wrapper(attr['initialize'])

        return super(LayerMeta, cls).__new__(cls, name, bases, attr)


class Layer(object, metaclass=LayerMeta):

    sep = '.'

    def __init__(self):
        self._name = self.__class__.__name__
        self._unique_name = None
        self._initialized = False
        self._parent = None
        self._layers = dict()
        self.param_names = []
        self.state_names = []

    def initialize(self, *input):
        pass

    def forward(self, *input):
        pass

    def __call__(self, *args, **kwargs):
        if not self._initialized:
            self.initialize(*args, **kwargs)
            self._initialized = True

        return self.forward(*args, **kwargs)

    def get_params(self):
        params = dict()
        sublayers = self._layers
        prefix = self._get_unique_name() + Layer.sep
        for name, sublayer in sublayers.items():
            params.update(sublayer.get_params())
        for param_name in self.param_names:
            params[prefix + param_name] = self.__dict__[param_name]
        return params

    def set_params(self, **parameters):
        # set parameters for Layer
        # input should be either a PyTensor or numpy ndarray.
        # examples: Layer.set_params(W=np.ones((in, out), dtype=np.float32)),
        # Layer.set_params(**{'block1':{'linear1':{'W':np.ones((in, out),
        # dtype=np.float32)}}})
        for (name, param) in list(parameters.items()):
            # assert isinstance(self.__dict__[parameter_name], Layer)
            if name.find(Layer.sep) < 0:
                if name in self.__dict__:
                    self.set_attribute(name, param, self.param_names)
                elif name in self._layers:
                    self._layers[name].set_params(**param)
                else:
                    raise ValueError("please input correct parameters.")
                del parameters[name]

        sublayers = self._layers
        prefix = self._get_unique_name() + Layer.sep
        for name in self.param_names:
            key = prefix + name
            if key in parameters:
                self.set_attribute(name, parameters[key], self.param_names)
        for name, sublayer in sublayers.items():
            sublayer.set_params(**parameters)

    def get_states(self):
        states = dict()
        sublayers = self._layers
        prefix = self._get_unique_name() + Layer.sep
        for name, sublayer in sublayers.items():
            states.update(sublayer.get_states())
        for state_name in self.state_names:
            states[prefix + state_name] = self.__dict__[state_name]
        return states

    def set_states(self, **states):
        for (name, state) in list(states.items()):
            if name.find(Layer.sep) < 0:
                if name in self.__dict__:
                    self.set_attribute(name, state, self.state_names)
                elif name in self._layers:
                    self._layers[name].set_params(**state)
                else:
                    raise ValueError("please input correct states.")
                del states[name]

        sublayers = self._layers
        prefix = self._get_unique_name() + Layer.sep
        for name in self.state_names:
            key = prefix + name
            if key in states:
                self.set_attribute(name, states[key], self.state_names)
        for name, sublayer in sublayers.items():
            sublayer.set_states(**states)

    def device_check(self, *inputs):
        x_device = inputs[0].device
        x_dev_id = x_device.id()
        for var in inputs:
            if var.device.id() != x_dev_id:
                var.to_device(x_device)

    def set_attribute(self, attribute_name, attribute_value, allow_attributes):
        assert (attribute_name in allow_attributes
               ), "please input allowed attributes."
        assert (attribute_value.shape == self.__dict__[attribute_name].shape
               ), "Shape dismatched."
        if isinstance(attribute_value, Tensor):
            self.__dict__[attribute_name].reset_like(attribute_value)
            self.__dict__[attribute_name].copy_data(attribute_value)
        elif isinstance(attribute_value, np.ndarray):
            self.__dict__[attribute_name].copy_from_numpy(attribute_value)
        else:
            raise ValueError("attributes should be Tensor or Numpy array.")

    def _get_unique_name(self):
        if not self._unique_name:
            prefix = ''
            if self._parent:
                prefix = self._parent._get_unique_name()
                if prefix:
                    prefix += Layer.sep

            self.__dict__['_unique_name'] = prefix + self._name

        return self._unique_name

    def __getattr__(self, name):
        if '_layers' in self.__dict__:
            layers = self.__dict__['_layers']
            if name in layers:
                return layers[name]
        object.__getattr__(self, name)

    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            # TODO: remove the attr from dict first
            self.__dict__['_layers'][name] = value
            value.__dict__['_parent'] = self
            value.__dict__['_name'] = name
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._layers:
            del self._layers[name]
        else:
            object.__delattr__(self, name)


class Linear(Layer):
    """
    Generate a Linear operator
    """

    # TODO: replace current with
    #   def __init__(self, out_features, bias=True):
    def __init__(self, out_features, *args, bias=True, **kwargs):
        """
        Args:
            out_channels: int, the channel of output, also is the number of
                filters
            bias: bool
        """
        super(Linear, self).__init__()

        self.out_features = out_features

        # TODO: for backward compatibility, to remove
        if len(args) > 0:
            self.in_features = out_features
            self.out_features = args[0]
        if len(args) > 1:
            self.bias = args[1]
        else:
            self.bias = bias

        if self.bias:
            self.param_names = ['W', 'b']
        else:
            self.param_names = ['W']
        self.state_names = self.param_names

    def initialize(self, x):
        self.in_features = x.shape[1]
        w_shape = (self.in_features, self.out_features)
        b_shape = (self.out_features,)

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape, requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)
        else:
            self.b = None

    def forward(self, x):
        if self.b:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)

        assert x.shape[1] == self.W.shape[0], (
            "Linear layer expects input features size %d received %d" %
            (self.W.shape[0], x.shape[1]))

        y = autograd.matmul(x, self.W)
        if self.bias:
            y = autograd.add_bias(y, self.b, axis=0)
        return y


class Gemm(Layer):
    """
    Generate a Gemm operator
    Y = alpha * A' * B' + beta * C
    B is weight, C is bias
    """

    def __init__(self,
                 out_features,
                 alpha=1.0,
                 beta=1.0,
                 transA=False,
                 transB=True,
                 bias=True):
        """
        Args:
            out_channels: int, the channel of output, also is the number of
                filters
            alpha (float): Scalar multiplier for the product of input tensors A * B.
            beta (float): Scalar multiplier for input tensor C.
            ransA (bool): Whether A should be transposed
            transB (bool): Whether B should be transposed
            bias: bool
        """
        super(Gemm, self).__init__()
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.transA = 1 if transA else 0
        self.transB = 1 if transB else 0
        self.bias = bias

        if self.bias:
            self.param_names = ['W', 'b']
        else:
            self.param_names = ['W']
        self.state_names = self.param_names

    def initialize(self, x):
        if self.transA == 0:
            self.in_features = x.shape[1]
        else:
            self.in_features = x.shape[0]

        if self.transB == 0:
            w_shape = (self.in_features, self.out_features)
        else:
            w_shape = (self.out_features, self.in_features)
        b_shape = (1, self.out_features)

        self.W = Tensor(shape=w_shape,
                        requires_grad=True,
                        stores_grad=True,
                        device=x.device)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape,
                            requires_grad=True,
                            stores_grad=True,
                            device=x.device)
            self.b.set_value(0.0)
        else:
            self.b = None

    def forward(self, x):
        if self.b:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)

        if self.transA == 0:
            in_features = x.shape[1]
        else:
            in_features = x.shape[0]

        if self.transB == 0:
            in_features_w = self.W.shape[0]
        else:
            in_features_w = self.W.shape[1]

        assert in_features == in_features_w, (
            "Gemm layer expects input features size %d received %d" %
            (in_features_w, in_features))
        y = autograd.gemm(x, self.W, self.b, self.alpha, self.beta, self.transA,
                          self.transB)

        return y


class Conv2d(Layer):
    """
    Generate a Conv 2d operator
    """

    def __init__(self,
                 nb_kernels,
                 kernel_size,
                 *args,
                 stride=1,
                 padding=0,
                 dilation=1,
                 group=1,
                 bias=True,
                 pad_mode="NOTSET",
                 **kwargs):
        """
        Args:
            nb_kernels (int): the channel of output, also is the number of filters
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            dilation (int): only support 1
            group (int): group
            bias (bool): bias
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(Conv2d, self).__init__()

        # the old code create the layer like: Conv2d(8, 16, 3)， or Conv2d(8, 16, 3, stride=1)
        # the following code block is for backward compatibility
        if len(args) > 0:
            nb_kernels = kernel_size
            kernel_size = args[0]
        if len(args) > 1:
            stride = args[1]
        if len(args) > 2:
            padding = args[2]

        self.nb_kernels = nb_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.bias = bias
        self.pad_mode = pad_mode

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise TypeError("Wrong stride type.")

        self.odd_padding = (0, 0, 0, 0)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) == 2:
                self.padding = padding
            elif len(padding) == 4:
                _h_mask = padding[0] - padding[1]
                _w_mask = padding[2] - padding[3]
                # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
                # so we need to firstly handle the input, then use the nomal padding method.
                self.odd_padding = (max(_h_mask, 0), max(-_h_mask, 0),
                                    max(_w_mask, 0), max(-_w_mask, 0))
                self.padding = (
                    padding[0] - self.odd_padding[0],
                    padding[2] - self.odd_padding[2],
                )
            else:
                raise TypeError("Wrong padding value.")

        if dilation != 1:
            raise ValueError("Not implemented yet")

        self.inner_params = {
            "cudnn_prefer": "fastest",
            "workspace_MB_limit": 1024,
        }
        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in self.inner_params:
                raise TypeError("Keyword argument not understood:", kwarg)
            else:
                self.inner_params[kwarg] = kwargs[kwarg]

        if self.bias:
            self.param_names = ['W', 'b']
        else:
            self.param_names = ['W']
        self.state_names = self.param_names

    def initialize(self, x):
        self.in_channels = x.shape[1]
        w_shape = (
            self.nb_kernels,
            int(self.in_channels / self.group),
            self.kernel_size[0],
            self.kernel_size[1],
        )

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        # std = math.sqrt(
        # 2.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1] +
        # self.nb_kernels))
        std = math.sqrt(
            2.0 / (w_shape[1] * self.kernel_size[0] * self.kernel_size[1] +
                   self.nb_kernels))
        self.W.gaussian(0.0, std)

        if self.bias:
            b_shape = (self.nb_kernels,)
            self.b = Tensor(shape=b_shape, requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)
        else:
            # to keep consistency when to do forward.
            self.b = None
            # Tensor(data=CTensor([]), requires_grad=False, stores_grad=False)

        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)

        if self.odd_padding != (0, 0, 0, 0):
            x = x.clone()
            x.data = utils.handle_odd_pad_fwd(x.data, self.odd_padding)

        if x.device.id() == -1:
            if self.group != 1:
                raise ValueError("Not implemented yet")
            else:
                if (not hasattr(self, "handle")) or (x.shape[0] !=
                                                     self.handle.batchsize):
                    self.handle = singa.ConvHandle(
                        x.data,
                        self.kernel_size,
                        self.stride,
                        self.padding,
                        self.in_channels,
                        self.nb_kernels,
                        self.bias,
                        self.group,
                    )
        else:
            if (not hasattr(self,
                            "handle")) or (x.shape[0] != self.handle.batchsize):
                self.handle = singa.CudnnConvHandle(
                    x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.in_channels,
                    self.nb_kernels,
                    self.bias,
                    self.group,
                )


    def forward(self, x):
        assert (self.group >= 1 and self.in_channels %
                self.group == 0), "please set reasonable group."

        assert (self.nb_kernels >= self.group and self.nb_kernels %
                self.group == 0), "nb_kernels and group dismatched."

        y = autograd.conv2d(self.handle, x, self.W, self.b, self.odd_padding)
        return y


class SeparableConv2d(Layer):
    """
    Generate a Conv 2d operator
    """

    def __init__(self,
                 nb_kernels,
                 kernel_size,
                 *args,
                 stride=1,
                 padding=0,
                 bias=False):
        """
        Args:
            nb_kernels (int): the channel of output, also is the number of filters
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            bias (bool): bias
        """
        super(SeparableConv2d, self).__init__()

        # the following code block is for backward compatibility
        if len(args) > 0:
            nb_kernels = kernel_size
            kernel_size = args[0]
        if len(args) > 1:
            stride = args[1]
        if len(args) > 2:
            padding = args[2]

        self.nb_kernels = nb_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def initialize(self, x):
        self.in_channels = x.shape[1]
        self.depthwise_conv = Conv2d(
            self.in_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            group=self.in_channels,
            bias=self.bias,
        )

        self.point_conv = Conv2d(self.nb_kernels, 1, bias=self.bias)

    def forward(self, x):
        y = self.depthwise_conv(x)
        y = self.point_conv(y)
        return y


class BatchNorm2d(Layer):
    """
    Generate a BatchNorm 2d operator
    """

    def __init__(self, momentum=0.9):
        """
        Args:
            momentum (float): Factor used in computing the running mean and
                variance.
        """
        super(BatchNorm2d, self).__init__()

        self.momentum = momentum

        self.param_names = ['scale', 'bias']
        self.state_names = self.param_names + ['running_mean', 'running_var']

    def initialize(self, x):
        self.channels = x.shape[1]
        param_shape = (self.channels,)

        self.scale = Tensor(shape=param_shape,
                            requires_grad=True,
                            stores_grad=True)
        self.scale.set_value(1.0)

        self.bias = Tensor(shape=param_shape,
                           requires_grad=True,
                           stores_grad=True)
        self.bias.set_value(0.0)

        self.running_mean = Tensor(shape=param_shape,
                                   requires_grad=False,
                                   stores_grad=False)
        self.running_mean.set_value(0.0)

        self.running_var = Tensor(shape=param_shape,
                                  requires_grad=False,
                                  stores_grad=False)
        self.running_var.set_value(1.0)

        if not hasattr(self, "handle"):
            if x.device.id() == -1:
                self.handle = singa.BatchNormHandle(self.momentum, x.data)
            else:
                self.handle = singa.CudnnBatchNormHandle(self.momentum, x.data)

    def forward(self, x):
        assert x.shape[1] == self.channels, (
            "number of channels dismatched. %d vs %d" %
            (x.shape[1], self.channels))

        self.device_check(x, self.scale, self.bias, self.running_mean,
                          self.running_var)

        y = autograd.batchnorm_2d(
            self.handle,
            x,
            self.scale,
            self.bias,
            self.running_mean,
            self.running_var,
        )
        return y


class Pooling2d(Layer):
    """
    Generate a Pooling 2d operator
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 is_max=True,
                 pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            is_max (bool): is max pooling or avg pooling
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(Pooling2d, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
            assert stride[0] > 0 or (kernel_size[0] == 1 and padding[0] == 0), (
                "stride[0]=0, but kernel_size[0]=%d, padding[0]=%d" %
                (kernel_size[0], padding[0]))
        else:
            raise TypeError("Wrong stride type.")

        self.odd_padding = (0, 0, 0, 0)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) == 2:
                self.padding = padding
            elif len(padding) == 4:
                _h_mask = padding[0] - padding[1]
                _w_mask = padding[2] - padding[3]
                # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
                # so we need to firstly handle the input, then use the nomal padding method.
                self.odd_padding = (max(_h_mask, 0), max(-_h_mask, 0),
                                    max(_w_mask, 0), max(-_w_mask, 0))
                self.padding = (
                    padding[0] - self.odd_padding[0],
                    padding[2] - self.odd_padding[2],
                )
            else:
                raise TypeError("Wrong padding value.")

        self.is_max = is_max
        self.pad_mode = pad_mode

    def initialize(self, x):
        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)

        out_shape_h = (int(
            (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) //
            self.stride[0]) + 1)
        out_shape_w = (int(
            (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) //
            self.stride[1]) + 1)

        if x.device.id() == -1:
            self.handle = singa.PoolingHandle(
                x.data,
                self.kernel_size,
                self.stride,
                self.padding,
                self.is_max,
            )
        else:
            self.handle = singa.CudnnPoolingHandle(
                x.data,
                self.kernel_size,
                self.stride,
                self.padding,
                self.is_max,
            )

    def forward(self, x):
        y = autograd.pooling_2d(self.handle, x, self.odd_padding)
        return y


class MaxPool2d(Pooling2d):
    """
    Generate a Max Pooling 2d operator
    """

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, True,
                                        pad_mode)


class AvgPool2d(Pooling2d):

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, False,
                                        pad_mode)


class MaxPool1d(Pooling2d):
    """
    Generate a Max Pooling 1d operator
    """

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        if stride is None:
            stride = kernel_size
        super(MaxPool1d, self).__init__((1, kernel_size), (1, stride),
                                        (0, padding), True, pad_mode)


class AvgPool1d(Pooling2d):
    """
    Generate a Avg Pooling 1d operator
    """

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        if stride is None:
            stride = kernel_size
        super(AvgPool1d, self).__init__((1, kernel_size), (1, stride),
                                        (0, padding), False, pad_mode)


class RNN_Base(Layer):

    def step_forward(self,
                     x=None,
                     h=None,
                     c=None,
                     Wx=None,
                     Wh=None,
                     Bx=None,
                     Bh=None,
                     b=None):
        raise NotImplementedError


class RNN(RNN_Base):
    """
    Generate a RNN operator
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """
        Args:
            input_size (int):  The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int):  Number of recurrent layers. Default: 1
            nonlinearity (string): The non-linearity to use. Default: 'tanh'
            bias (bool):  If False, then the layer does not use bias weights.
                Default: True
            batch_first (bool):  If True, then the input and output tensors
                are provided as (batch, seq, feature). Default: False
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each RNN layer except the last layer, with dropout
                probability equal to dropout. Default: 0
            bidirectional (bool): If True, becomes a bidirectional RNN.
                Default: False
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def initialize(self, xs, h0):
        Wx_shape = (self.input_size, self.hidden_size)
        self.Wx = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx.gaussian(0.0, 1.0)

        Wh_shape = (self.hidden_size, self.hidden_size)
        self.Wh = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh.gaussian(0.0, 1.0)

        B_shape = (self.hidden_size,)
        self.b = Tensor(shape=B_shape, requires_grad=True, stores_grad=True)
        self.b.set_value(0.0)

        self.param_names = ['Wx', 'Wh', 'b']
        self.state_names = self.param_names

    def forward(self, xs, h0):
        # xs: a tuple or list of input tensors
        if not isinstance(xs, tuple):
            xs = tuple(xs)
        inputs = xs + (h0,)
        self.device_check(*inputs)
        # self.device_check(inputs[0], *self.params)
        self.device_check(inputs[0], self.Wx, self.Wh, self.b)
        batchsize = xs[0].shape[0]
        out = []
        h = self.step_forward(xs[0], h0, self.Wx, self.Wh, self.b)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h = self.step_forward(x, h, self.Wx, self.Wh, self.b)
            out.append(h)
        return out, h

    def step_forward(self, x, h, Wx, Wh, b):
        y2 = autograd.matmul(h, Wh)
        y1 = autograd.matmul(x, Wx)
        y = autograd.add(y2, y1)
        y = autograd.add_bias(y, b, axis=0)
        if self.nonlinearity == "tanh":
            y = autograd.tanh(y)
        elif self.nonlinearity == "relu":
            y = autograd.relu(y)
        else:
            raise ValueError
        return y


class LSTM(RNN_Base):
    """
    Generate a LSTM operator
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity="tanh",
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """
        Args:
            input_size (int):  The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int):  Number of recurrent layers. Default: 1
            nonlinearity (string): The non-linearity to use. Default: 'tanh'
            bias (bool):  If False, then the layer does not use bias weights.
                Default: True
            batch_first (bool):  If True, then the input and output tensors
                are provided as (batch, seq, feature). Default: False
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each RNN layer except the last layer, with dropout
                probability equal to dropout. Default: 0
            bidirectional (bool): If True, becomes a bidirectional RNN.
                Default: False
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def initialize(self, xs, h0_c0):
        # 1. Wx_i input,  Bx_i
        # 2. Wx_f forget, Bx_f
        # 3. Wx_o output, Bx_o
        # 4. Wx_g candidate, Bx_g
        Wx_shape = (self.input_size, self.hidden_size)
        self.Wx_i = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx_f = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx_o = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx_g = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)

        Wh_shape = (self.hidden_size, self.hidden_size)
        self.Wh_i = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh_f = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh_o = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh_g = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        [
            w.gaussian(0.0, 0.01) for w in [
                self.Wx_i, self.Wx_f, self.Wx_o, self.Wx_g, self.Wh_i,
                self.Wh_f, self.Wh_o, self.Wh_g
            ]
        ]

        Bx_shape = (self.hidden_size,)
        self.Bx_i = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bx_f = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bx_o = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bx_g = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_i = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_f = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_o = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        self.Bh_g = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
        [
            b.set_value(0.0) for b in [
                self.Bx_i, self.Bx_f, self.Bx_o, self.Bx_g, self.Bh_i,
                self.Bh_f, self.Bh_o, self.Bh_g
            ]
        ]

        self.param_names = [
            'Wx_i', 'Wx_f', 'Wx_o', 'Wx_g', 'Wh_i', 'Wh_f', 'Wh_o', 'Wh_g',
            'Bx_i', 'Bx_f', 'Bx_o', 'Bx_g', 'Bh_i', 'Bh_f', 'Bh_o', 'Bh_g'
        ]
        self.state_names = self.param_names

    def forward(self, xs, h0_c0):
        # xs: a tuple or list of input tensors
        # h0_c0: a tuple of (h0, c0)
        h0, c0 = h0_c0
        if not isinstance(xs, list):
            xs = list(xs)
        inputs = xs + list((h0, c0))
        self.device_check(*inputs)
        self.device_check(
            inputs[0],
            *[self.__dict__[param_name] for param_name in self.param_names])
        batchsize = xs[0].shape[0]
        out = []
        h, c = self.step_forward(xs[0], h0, c0)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h, c = self.step_forward(x, h, c)
            out.append(h)
        return out, h, c

    def step_forward(self, x, h, c):
        # input
        y1 = autograd.matmul(x, self.Wx_i)
        y1 = autograd.add_bias(y1, self.Bx_i, axis=0)
        y2 = autograd.matmul(h, self.Wh_i)
        y2 = autograd.add_bias(y2, self.Bh_i, axis=0)
        i = autograd.add(y1, y2)
        i = autograd.sigmoid(i)

        # forget
        y1 = autograd.matmul(x, self.Wx_f)
        y1 = autograd.add_bias(y1, self.Bx_f, axis=0)
        y2 = autograd.matmul(h, self.Wh_f)
        y2 = autograd.add_bias(y2, self.Bh_f, axis=0)
        f = autograd.add(y1, y2)
        f = autograd.sigmoid(f)

        # output
        y1 = autograd.matmul(x, self.Wx_o)
        y1 = autograd.add_bias(y1, self.Bx_o, axis=0)
        y2 = autograd.matmul(h, self.Wh_o)
        y2 = autograd.add_bias(y2, self.Bh_o, axis=0)
        o = autograd.add(y1, y2)
        o = autograd.sigmoid(o)

        y1 = autograd.matmul(x, self.Wx_g)
        y1 = autograd.add_bias(y1, self.Bx_g, axis=0)
        y2 = autograd.matmul(h, self.Wh_g)
        y2 = autograd.add_bias(y2, self.Bh_g, axis=0)
        g = autograd.add(y1, y2)
        g = autograd.tanh(g)

        cout1 = autograd.mul(f, c)
        cout2 = autograd.mul(i, g)
        cout = autograd.add(cout1, cout2)

        hout = autograd.tanh(cout)
        hout = autograd.mul(o, hout)
        return hout, cout


''' import autograd at the end to resolve circular import
'''
from singa import autograd
