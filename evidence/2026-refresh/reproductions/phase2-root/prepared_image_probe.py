import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from plato.config import Config
from plato.datasources import cinic10, tiny_imagenet
from tests.integration.utils import build_minimal_config, configure_environment

results=[]
with tempfile.TemporaryDirectory(prefix='plato-prepared-images-') as temporary:
    for name, module in [('tiny',tiny_imagenet),('cinic',cinic10)]:
        root=Path(temporary)/name
        root.mkdir()
        with configure_environment(build_minimal_config(),runtime_root=root):
            data=Path(Config().params['data_path'])
            for part, cls in [('train','n001'),('train','n002'),('test','n002')]:
                dest=data/part/cls/'one.png'
                dest.parent.mkdir(parents=True,exist_ok=True)
                rows,cols=np.indices((64,64))
                pixels=np.stack((rows*4,cols*4,(rows+cols)*2),axis=2).astype(np.uint8)
                Image.fromarray(pixels).save(dest)
            source=module.DataSource()
            train=source.get_train_set();test=source.get_test_set()
            torch.manual_seed(27)
            first=test[0][0];second=test[0][0]
            results.append({'dataset':name,'train_classes':train.classes,'test_classes':test.classes,'actual_test_targets':test.targets,'expected_training_space_target':train.class_to_idx['n002'],'repeated_test_equal':bool(torch.equal(first,second)),'max_abs_test_difference':float((first-second).abs().max())})
print('PROBE_RESULT',json.dumps(results))
