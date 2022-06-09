import torch
import os
import math

from plato.config import Config
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from torchvision import transforms

tt = transforms.ToPILImage()


def main():
    result_path = Config().results.result_path
    result_file = f"{result_path}/plot.png"
    try:
        load_file = os.path.join(result_path, [
            file for file in os.listdir(result_path)
            if (file.endswith("tensors.pt"))
        ][0])
        history_dict = torch.load(load_file)
    except:
        print("File failed to load")
        exit()

    num_images = len(history_dict[0])

    fig = plt.figure(figsize=(12, 8))
    rows = math.ceil(len(history_dict) / 2)
    outer = gridspec.GridSpec(rows, 2, wspace=0.2, hspace=0.2)

    for count, item in enumerate(history_dict.items()):
        # item[0] holds the iteration number
        # item[1] holds a dictionary of PIL Images for each reconstructed image
        inner = gridspec.GridSpecFromSubplotSpec(1,
                                                 num_images,
                                                 subplot_spec=outer[count])
        outerplot = plt.Subplot(fig, outer[count])
        outerplot.set_title("Iter=%d" % item[0])
        outerplot.axis('off')
        fig.add_subplot(outerplot)

        for img_num in range(num_images):
            innerplot = plt.Subplot(fig, inner[img_num])
            innerplot.imshow(item[1][img_num])
            innerplot.axis('off')
            fig.add_subplot(innerplot)
    fig.savefig(result_file)


if __name__ == "__main__":
    main()