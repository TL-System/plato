import os

os.system('rm ' + park.__path__[0] + '/envs/abr/ipc_msg_pb2.py')
os.system('rm ' + park.__path__[0] + '/envs/abr/ipc_msg_pb2.pyc')
os.system('protoc -I=' + park.__path__[0] + '/envs/abr/' +
          '--python_out=' + park.__path__[0] + '/envs/abr/' +
          ' ' + park.__path__[0] + '/envs/abr/ipc_msg.proto')
