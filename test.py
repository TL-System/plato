import mindspore.nn as nn
import mindspore as ms

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor

# 1. Build a dataset.
download_train = Mnist(path="./mnist", split="train", batch_size=batch_size, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

# 2. Define a neural network.
network = lenet(num_classes=10, pretrained=False)
# 3.1 Define a loss function.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 3.2 Define an optimizer function.
net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=momentum)
# 3.3 Initialize model parameters.
model = ms.Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})

# 4. Train the neural network.
model.train(epochs, dataset_train, callbacks=[LossMonitor(learning_rate, 1875)])

