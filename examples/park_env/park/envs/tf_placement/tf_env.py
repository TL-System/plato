import numpy as np
import tensorflow as tf
from time import time
from park.envs.tf_placement.model import get_model

class TFRuntime(object):

  def __init__(self, model, devs):
    g, trainer = get_model(model, devs, return_train_op=True)
    self.g = g
    self.train_op = trainer
    self.ops = self.g.get_operations()

  def measure(self, pl):
    with self.g.as_default():

      for op in self.ops:
        if op.name in pl:
          dev = pl[op.name]
          op._set_device('/gpu:' + str(dev))

      with tf.train.MonitoredTrainingSession(
          config=tf.ConfigProto(allow_soft_placement=True,
              log_device_placement=False)) as sess:

          rts = []
          for _ in range(10):
            s_t = time()
            sess.run(self.train_op)
            e_t = time()
            rts.append(e_t - s_t)

      rt = np.average(rts[-5:])


    return float(rt) * 1e6 # microseconds
