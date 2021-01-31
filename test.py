import numpy as np
import mindspore.dataset as ds

np.random.seed(58)
data = np.random.sample((5, 2))
label = np.random.sample((5, 1))
feature_dataset = [(data, label), (data, label)]

def GeneratorFunc():
    for logit, target in feature_dataset:
        yield (logit, target)

dataset = ds.GeneratorDataset(GeneratorFunc, ["data", "label"])

print(dataset.get_dataset_size())
for sample in dataset.create_dict_iterator():
    print(sample["data"], sample["label"])
