"""
Testing a federated learning configuration.

How to run the tests:

 1 Run the following command in the root directory.
    python tests/config_tests.py

"""

import os
import unittest
import warnings

from plato.config import Config

os.environ["config_file"] = "configs/Kinetics/kinetics_mm.yml"


class ConfigTest(unittest.TestCase):
    """Testing the correctness of loading a configuration file."""

    def setUp(self):
        super().setUp()

        # define several example parameters that will be used as
        # a demo to test the loaded configuration file.
        data_params_config = {
            "downloader": {"num_workers": 4},
            "multi_modal_configs": {
                "rgb": {"train": {"type": "RawframeDataset"}},
                "flow": {"train": {"type": "RawframeDataset"}},
                "audio": {"train": {"type": "AudioFeatureDataset"}},
            },
        }
        model_params_config = {
            "model_name": "rgb_flow_audio_model",
            "model_config": {"rgb_model": {"type": "Recognizer3D"}},
        }

        self.data_config = Config.namedtuple_from_dict(data_params_config)
        self.model_config = Config.namedtuple_from_dict(model_params_config)

    def test_dataconfig(self):
        """Test the structure and necessary parameters of the data configuration.

        This function tests the core part of the configuration file. It tests
        the following elements:

        downloader, num_workers, multi_modal_configs, and whether there are
        three modalities, including rgb, flow, and audio and the corresponding
        data types are correct.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            defined_data_config = Config().data

            hasattr(defined_data_config, "downloader")
            hasattr(defined_data_config.downloader, "num_workers")

            self.assertEqual(
                defined_data_config.downloader.num_workers,
                self.data_config.downloader.num_workers,
            )

            hasattr(defined_data_config, "multi_modal_configs")
            hasattr(defined_data_config.multi_modal_configs, "rgb")
            hasattr(defined_data_config.multi_modal_configs, "flow")
            hasattr(defined_data_config.multi_modal_configs, "audio")
            hasattr(defined_data_config.multi_modal_configs.rgb, "train")

            self.assertEqual(
                defined_data_config.multi_modal_configs.rgb.train.type,
                self.data_config.multi_modal_configs.rgb.train.type,
            )

    def test_modelconfig(self):
        """Test the structure and necessary parameters of the model configuration.

        This function tests the core part of the configuration file. It tests whether
        the model is defined and whether the configurations for the defined model is
        correct.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            defined_model_config = Config().parameters.model

            hasattr(defined_model_config, "model_name")

            self.assertEqual(defined_model_config.model_name, "rgb_flow_audio_model")
            self.assertEqual(
                defined_model_config.model_config.rgb_model.type,
                self.model_config.model_config.rgb_model.type,
            )


if __name__ == "__main__":
    unittest.main()
