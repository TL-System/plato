import torch
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


resnet18 = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
mobilenet = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
alexnet = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)

print("The size of ResNet-18:")
count_parameters(resnet18)

print("\nThe size of MobileNet:")
count_parameters(mobilenet)

print("\nThe size of AlexNet:")
count_parameters(alexnet)
