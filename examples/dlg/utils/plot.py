import math
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from plato.config import Config
from torchvision import transforms

tt = transforms.ToPILImage()


def main():
    result_path = Config().results.result_path
    result_file = f"{result_path}/{os.getpid()}_plot.png"
    try:
        # Gets all the directories for each individual run
        subprocesses = [
            file
            for file in os.listdir(result_path)
            if (os.path.isdir(os.path.join(result_path, file)))
        ]
        if hasattr(Config().results, "subprocess"):
            if str(Config().results.subprocess) in subprocesses:
                subprocess_path = os.path.join(
                    result_path, str(Config().results.subprocess)
                )
            else:
                print("Subprocess not found")
                exit()
        else:
            # Select the latest run
            subprocess_path = os.path.join(result_path, subprocesses[0])

        trials = [
            file
            for file in os.listdir(subprocess_path)
            if (os.path.isdir(os.path.join(subprocess_path, file)))
        ]

        if hasattr(Config().results, "trial"):
            trial = f"t{Config().results.trial}"
            if trial not in trials:
                print("Trial not found")
                exit()
        else:
            trial = [file_name for file_name in trials if "best" in file_name][0]

        final_path = os.path.join(subprocess_path, trial)
        tensor_file = f"{final_path}/tensors.pt"
        print("loading", tensor_file)
        history_dict = torch.load(tensor_file)
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
        inner = gridspec.GridSpecFromSubplotSpec(
            1, num_images, subplot_spec=outer[count]
        )
        outerplot = plt.Subplot(fig, outer[count])
        outerplot.set_title("Iter=%d" % item[0])
        outerplot.axis("off")
        fig.add_subplot(outerplot)

        for img_num in range(num_images):
            innerplot = plt.Subplot(fig, inner[img_num])
            innerplot.imshow(item[1][img_num])
            innerplot.axis("off")
            fig.add_subplot(innerplot)
    fig.savefig(result_file)
    print("file saved to", result_file)


if __name__ == "__main__":
    main()
