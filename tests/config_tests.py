"""
Testing a federated learning configuration.
"""

import os
import unittest
import warnings

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.yml'

from plato.config import Config


class ConfigTest(unittest.TestCase):
    """ Testing the correctness of loading a configuration file. """
    def setup(self):
        super().setup()

        self.addTypeEqualityFunc(Config, 'assertConfigEqual')

        self.defined_config = Config()

        # define several example parameters
        data_params_config = {
            "downloader": {
                "num_workers": 4
            },
            "multi_modal_pipeliner": {
                "rgb": {
                    "rgb_data": {
                        "train": {
                            "type": "RawframeDataset"
                        }
                    }
                },
                "flow": {
                    "flow_data": {
                        "train": {
                            "type": "RawframeDataset"
                        }
                    }
                },
                "audio": {
                    "audio_data": {
                        "train": {
                            "type": "AudioFeatureDataset"
                        }
                    }
                }
            }
        }
        model_params_config = {
            "model_name": "rgb_flow_audio_model",
            "model_config": {
                "rgb_model": {
                    "type": "Recognizer3D"
                }
            }
        }

        self.data_config = Config.namedtuple_from_dict(data_params_config)
        self.model_config = Config.namedtuple_from_dict(model_params_config)

    def assertAttrContained(self, src_config, dst_key):
        """ The attrs of dst_key should be included in the src_config"""
        assert hasattr(src_config, dst_key)

    def assertAttrValueEqual(self, src_value, dst_value):
        """ The two values should be equal with each other """
        self.assertEqual(src_value, dst_value)

    def test_dataconfig(self):
        """ Test the structure and necessary parameters of the data configuration """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            defined_data_config = Config().data

            self.assertAttrContained(src_config=defined_data_config,
                                     dst_key="downloader")
            self.assertAttrContained(src_config=defined_data_config.downloader,
                                     dst_key="num_workers")
            self.assertAttrValueEqual(
                src_value=defined_data_config.downloader.num_workers,
                dst_value=self.data_config.downloader.num_workers)

            self.assertAttrContained(src_config=defined_data_config,
                                     dst_key="multi_modal_pipeliner")
            self.assertAttrContained(
                src_config=defined_data_config.multi_modal_pipeliner,
                dst_key="rgb")
            self.assertAttrContained(
                src_config=defined_data_config.multi_modal_pipeliner,
                dst_key="flow")
            self.assertAttrContained(
                src_config=defined_data_config.multi_modal_pipeliner,
                dst_key="audio")

            self.assertAttrContained(
                src_config=defined_data_config.multi_modal_pipeliner.rgb,
                dst_key="rgb_data")
            self.assertAttrValueEqual(
                src_value=defined_data_config.multi_modal_pipeliner.rgb.
                rgb_data.train.type,
                dst_value=self.data_config.multi_modal_pipeliner.rgb.rgb_data.
                train.type)

    def test_modelconfig(self):
        """ Test the structure and necessary parameters of the model configuration """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            defined_model_config = Config().model
            self.assertAttrContained(src_config=defined_model_config,
                                     dst_key="model_name")
            self.assertAttrValueEqual(
                src_value=defined_model_config.model_name,
                dst_value="rgb_flow_audio_model")
            self.assertAttrValueEqual(
                src_value=defined_model_config.model_config.rgb_model.type,
                dst_value=self.model_config.model_config.rgb_model.type)


if __name__ == '__main__':
    unittest.main()
