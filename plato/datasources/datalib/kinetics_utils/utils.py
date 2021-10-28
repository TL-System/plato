

import os
import json


def extract_data_classes(data_classes_file, train_info_data_path,
                         val_info_data_path):
    """ Obtain a list of class names in the dataset. """

    classes_container = list()
    if os.path.exists(data_classes_file):
        with open(data_classes_file, "r") as class_file:
            lines = class_file.readlines()
            classes_container = [line.replace("\n", "") for line in lines]

        return classes_container

    if not os.path.exists(train_info_data_path) or not os.path.exists(
            val_info_data_path):
        logging.info(
            "The json files of the dataset are not completed. Download it first."
        )
        sys.exit()

    for list_path in [train_info_data_path, val_info_data_path]:
        with open(list_path) as file:
            videos_data = json.load(file)
        for key in videos_data.keys():
            metadata = videos_data[key]
            annotations = metadata["annotations"]
            label = annotations["label"]
            class_name = label.replace("_", " ")
            if class_name not in classes_container:
                classes_container.append(class_name)
    with open(data_classes_file, "w") as file:
        for class_name in classes_container:
            file.write(class_name)
            file.write('\n')

    return classes_container