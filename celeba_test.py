import torch
from plato.datasources import celeba

if __name__ == '__main__':
    ds = celeba.DataSource()
    total_num = 0
    train_loader = torch.utils.data.DataLoader(ds.trainset, batch_size=1)
    test_loader = torch.utils.data.DataLoader(ds.testset, batch_size=1)

    all_data = None
    for batch_id, (examples, labels) in enumerate(train_loader):
        if all_data is None:
            all_data = examples
        else:
            all_data = torch.cat((all_data, examples), 0)

    for batch_id, (examples, labels) in enumerate(test_loader):
        all_data = torch.cat((all_data, examples), 0)

    print(all_data.shape)
    print(torch.mean(all_data, [0, 2, 3]))
    print(torch.std(all_data, [0, 2, 3]))
