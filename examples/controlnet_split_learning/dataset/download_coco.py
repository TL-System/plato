"""Download the coco dataset."""
import sys
import os
import zipfile
import wget


def main():
    "The main function."
    path = sys.argv[1]
    if not os.path.exists(os.path.join(path, "train2017")):
        wget.download("http://images.cocodataset.org/zips/train2017.zip", path)
        with zipfile.ZipFile(os.path.join(path, "train2017.zip"), "r") as zip_ref:
            zip_ref.extractall()
    if not os.path.exists(os.path.join(path, "val2017")):
        wget.download("http://images.cocodataset.org/zips/val2017.zip", path)
        with zipfile.ZipFile(os.path.join(path, "train2017.zip"), "r") as zip_ref:
            zip_ref.extractall()
    if not os.path.exists(
        os.path.join(path, "annotations/captions_train2017.json")
    ) or os.path.exists(os.path.join(path, "annotations/captions_val2017.json")):
        wget.download(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            path,
        )
        with zipfile.ZipFile(
            os.path.join(path, "annotations_trainval2017.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall()
    if not os.path.exists(
        os.path.join(path, "annotations/stuff_train2017_pixelmaps")
    ) or os.path.exists(os.path.join(path, "annotations/stuff_val2017_pixelmaps")):
        wget.download(
            "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
            path,
        )
        with zipfile.ZipFile(
            os.path.join(path, "annotations/stuff_train2017_pixelmaps.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall()
        with zipfile.ZipFile(
            os.path.join(path, "annotations/stuff_val2017_pixelmaps.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall()


if __name__ == "__main__":
    main()
