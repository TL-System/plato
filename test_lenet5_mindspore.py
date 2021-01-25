import mindspore.nn as nn
import mindspore
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from datasets import registry as datasets_registry
from models import registry as models_registry
from trainers.trainer_mindspore import Trainer
from dividers.iid_mindspore import IIDDivider
from config import Config


def test_net(network, network_model):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    param_dict = {}
    for _, param in network.parameters_and_names():
        param_dict[param.name] = param  # load parameter to the network
    mindspore.load_param_into_net(network, param_dict)
    # load testing dataset
    # dataset = datasets_registry.get()
    ds_eval = dataset.get_test_set()

    acc = network_model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


if __name__ == "__main__":
    mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,
                                  device_target='GPU')

    # learning rate setting
    lr = 0.01
    momentum = 0.9
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_epoch = 1
    # create the network
    model_name = Config().trainer.model
    net = models_registry.get(model_name)
    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)

    dataset = datasets_registry.get()
    iid = IIDDivider(dataset)
    ds_train = iid.get_partition(partition_size=60000, client_id=1)

    count = 0
    for item in ds_train.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} batches".format(count))

    trainer = Trainer(net)

    weights = trainer.extract_weights()

    net2 = models_registry.get(model_name)
    trainer2 = Trainer(net2)
    trainer2.load_weights(weights)
    net2_opt = nn.Momentum(trainer2.model.trainable_params(), lr, momentum)

    # group layers into an object with training and evaluation features
    model = mindspore.Model(trainer2.model,
                            net_loss,
                            net2_opt,
                            metrics={"Accuracy": Accuracy()})

    # print(weights['conv1.weight'].asnumpy())

    model.train(train_epoch,
                ds_train,
                callbacks=[LossMonitor(per_print_times=300)])

    print("Second time")
    # Second
    ds2_train = iid.get_partition(partition_size=60000, client_id=1)

    trainer2.load_weights(trainer2.extract_weights())
    net2_opt = nn.Momentum(trainer2.model.trainable_params(), lr, momentum)

    model = mindspore.Model(trainer2.model,
                            net_loss,
                            net2_opt,
                            metrics={"Accuracy": Accuracy()})

    model.train(train_epoch,
                ds2_train,
                callbacks=[LossMonitor(per_print_times=300)])

    test_net(trainer2.model, model)
