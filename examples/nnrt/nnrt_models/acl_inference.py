"""
The inference class for NNRT model.
"""

import acl

from nnrt_models.constant import ACL_MEMCPY_HOST_TO_DEVICE
from nnrt_models.acl_model import Model
from nnrt_models.acl_util import check_ret


class Inference(object):
    """This class provide resources management and complete inference procedure. """
    def __init__(self, device_id, model_path, model_input_height,
                 model_input_width):

        self.device_id = device_id  # int
        self.model_path = model_path  # string
        self.model_id = None  # pointer
        self.context = None  # pointer

        self.input_data = None
        self.output_data = None
        self.model_desc = None  # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.init_resource()

        self._model_input_width = model_input_width
        self._model_input_height = model_input_height

        self.model_process = Model(self.context, self.stream, self.model_path)

    def __del__(self):
        """ Release preallocated resources. """
        if self.model_process:
            del self.model_process

        if self.stream:
            acl.rt.destroy_stream(self.stream)

        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    def init_resource(self):
        """ Allocate resouces for device context and stream. """
        acl.init()
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        print("[Sample] init resource stage success")

    def _transfer_to_device(self, input_img):
        """ Transfer input image from host to devicd. """
        img_ptr = acl.util.numpy_to_ptr(input_img)
        img_buffer_size = input_img.itemsize * input_img.size  # get byte size
        img_device, ret = acl.media.dvpp_malloc(img_buffer_size)
        check_ret("acl.media.dvpp_malloc", ret)
        ret = acl.rt.memcpy(img_device, img_buffer_size, img_ptr,
                            img_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)

        return img_device, img_buffer_size

    def forward(self, input_img):
        """ Pass the input image to model class for inference. """
        img_device, img_buffer_size = self._transfer_to_device(input_img)

        output = self.model_process.run(img_device, img_buffer_size)

        return output
