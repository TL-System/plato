"""
The NNRT-based client inference model, used by only client.

The OM model is precompiled and it can only deal with static input shape.
"""

import struct
import numpy as np
import acl

from nnrt_models.constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_DEVICE_TO_HOST, NPY_BYTE
from nnrt_models.acl_util import check_ret


class Model(object):
    """ This class provides resource management for inference model. """
    def __init__(
        self,
        context,
        stream,
        model_path,
    ):
        self.model_path = model_path
        self.model_id = None
        self.context = context
        self.stream = stream
        self.input_data = None
        self.output_data = None
        self.model_desc = None

        self.init_resource()

    def __del__(self):
        """ Release the preallocated resources. """
        self._release_dataset()
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

    def init_resource(self):
        """ Allocate resouces for inference model. """
        acl.rt.set_context(self.context)
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_output_dataset(output_size)

    def _gen_output_dataset(self, size):
        """ Create Output dataset buffer.
        size: The number of elements in input batch (minibatch size).
        """
        dataset = acl.mdl.create_dataset()
        for i in range(size):
            temp_buffer_size = acl.mdl.get_output_size_by_index(
                self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            dataset_buffer = acl.create_data_buffer(temp_buffer,
                                                    temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
            if ret:
                ret = acl.destroy_data_buffer(dataset_buffer)
                check_ret("acl.destroy_data_buffer", ret)
        self.output_data = dataset

    def run(self, input_buffer, input_size):
        self._gen_input_dataset(input_buffer, input_size)
        self.forward()
        result = self._get_result(self.output_data)

        return result

    def forward(self):
        """ Execute model from precompiled OM file. """
        ret = acl.mdl.execute(self.model_id, self.input_dataset,
                              self.output_data)
        check_ret("acl.mdl.execute", ret)

    def _gen_input_dataset(self, input_buffer, input_size):
        """ Create input dataset buffer for inference model.
        input_buffer: The memory holds the input data on device.
        input_size:   The size of device memory holding the input data.
        """
        self.input_dataset = acl.mdl.create_dataset()
        input_dataset_buffer = acl.create_data_buffer(input_buffer, input_size)
        _, ret = acl.mdl.add_dataset_buffer(self.input_dataset,
                                            input_dataset_buffer)
        if ret:
            ret = acl.destroy_data_buffer(input_dataset_buffer)
            check_ret("acl.destroy_data_buffer", ret)

    def _get_result(self, infer_output):
        """ Transfer output result from device to host and decode it as
        numpy array.
        infer_output: The output dataset buffer for inference model.
        """
        output = []
        num = acl.mdl.get_dataset_num_buffers(infer_output)
        for i in range(num):
            temp_output_buf = acl.mdl.get_dataset_buffer(infer_output, i)
            infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
            infer_output_size = acl.get_data_buffer_size_v2(temp_output_buf)
            output_host, _ = acl.rt.malloc_host(infer_output_size)
            acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
                          infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)
            result = acl.util.ptr_to_numpy(output_host, (infer_output_size, ),
                                           NPY_BYTE)
            "TODO: The unpack size depends on the cutlayer parameter. "
            result = struct.unpack("{:d}f".format(int(infer_output_size / 4)),
                                   bytearray(result))  # this is the ouput size
            output.append(result)
            ret = acl.rt.free_host(output_host)
            check_ret("acl.rt.free_host", ret)

        return np.array(output)

    def _release_dataset(self):
        """ Release dataset buffer for both input and output. """
        for dataset in [self.input_dataset, self.output_data]:
            if not dataset:
                continue

            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    data = acl.get_data_buffer_addr(data_buf)
                    ret = acl.rt.free(data)
                    check_ret("acl.rt.free", ret)
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)
