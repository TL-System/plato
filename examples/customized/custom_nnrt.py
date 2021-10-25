"""

"""

import os 
import sys
import numpy as np
sys.path.append("/var/lib/docker/ZSK/plato-main/plato-main")
sys.path.append("/var/lib/docker/ZSK/plato-main/plato-main/packages/yolov5")

from plato.clients.mistnet import Client
from plato.datasources.nnrt.yolo import DataSource
from plato.trainers.nnrt.yolo import Trainer 
from plato.algorithms.nnrt.mistnet import Algorithm
from plato.config import Config
from plato.models.nnrt.acl_inference import Inference

def main():
    """ A Plato mistnet training sesstion using a nnrt yolo model, datasource and trainer. """
    datasource = DataSource() # special datasource for yolo model

    model = Inference(int(Config().trainer.deviceID), 
                    Config().trainer.om_path, 
                    Config().data.input_height, 
                    Config().data.input_width)

    trainer = Trainer(model=model)
    algorithm = Algorithm(trainer)
    
    client = Client(model=model, datasource=datasource, 
                    algorithm=algorithm, trainer=trainer)
    client.load_data()
    _, features = client.train()

if __name__ == "__main__":
    main()
