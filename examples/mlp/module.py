#
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
#

from singa import module
from singa import autograd
from singa import tensor
from singa.tensor import Tensor


class MLP(module.Module):

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.dimension = 2

        self.linear1 = autograd.Linear(data_size, perceptron_size)
        self.linear2 = autograd.Linear(perceptron_size, num_classes)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = autograd.relu(x)
        x = self.linear2(x)
        return x

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss, dist_option, spars):
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = MLP(**kwargs)

    return model


__all__ = ['MLP', 'create_model']

if __name__ == "__main__":

    import numpy as np
    from singa import opt
    from singa import device

    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)
    # generate the training data
    x = np.random.uniform(-1, 1, 400)
    y = f(x) + 2 * np.random.randn(len(x))
    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

    dev = device.create_cuda_gpu_on(0)
    sgd = opt.SGD(0.05)
    tx = tensor.Tensor((400, 2), dev, tensor.float32)
    ty = tensor.Tensor((400,), dev, tensor.int32)
    model = MLP(data_size=2, perceptron_size=3, num_classes=2)

    # attached model to graph
    model.on_device(dev)
    model.set_optimizer(sgd)
    model.graph(True, False)
    model.train()

    for i in range(1001):
        tx.copy_from_numpy(data)
        ty.copy_from_numpy(label)
        out = model(tx)
        loss = model.loss(out, ty)
        model.optim(loss, 'fp32', spars=None)

        if i % 100 == 0:
            print("training loss = ", tensor.to_numpy(loss)[0])
