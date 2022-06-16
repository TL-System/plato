import sys
import tensorflow as tf

def get_inception(devs):
	from park.envs.tf_placement.models.inception.inception_model import InceptionV3
	bs = 64 
	g, trainer = InceptionV3().build(bs=bs,trainable=True, devices=devs)
	return g, trainer

def get_nasnet(devices, bs, model_level_placement):
	from models.nasnet.nasnets import Nasnet

	if bs is None:
			bs = 64 # default
	g, trainer = Nasnet(bs=bs, trainable=True, devices=devices,
											device_placement=model_level_placement)
	return g, trainer


def get_nmt(devices, bs, model_level_placement):
	from models.tf_seq2seq import train_toy
	# default already existing in train_toy argparse.
	# if bs is None:
	# 	bs = 256
	return train_toy.create_model(True, bs, None, devices=devices,
		device_placement=model_level_placement)

def get_model(model, devices, bs=None, model_level_placement=None,
							return_train_op=False, args=None):
	if model == 'inception':
		g, trainer = get_inception(devices)
	elif model == 'nasnet':
		g, train = get_nasnet(devices, bs, model_level_placement)
	elif model == 'nmt':
		g, trainer = get_nmt(devices, bs, model_level_placement)
	else:
		raise NotImplementedError('%s is not implemented'%model)

	if return_train_op:
		return g, trainer
		
	with g.as_default():
		return tf.train.export_meta_graph(clear_devices=False)
