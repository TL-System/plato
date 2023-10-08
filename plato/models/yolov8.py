"""
Obtaining a model from the Ultralytics.
"""

from ultralytics import YOLO

from plato.config import Config


class Model:
    """
    The model loaded from the YOLOv8.

    """

    @staticmethod
    # pylint: disable=unused-argument
    def get(model_name=None, **kwargs):
        """Returns the YOLOV8 model loaded from the Ultralytics."""
        model_type = Config().parameters.model.type

        return YOLO(model_type)
