import json
import numpy as np
import torch
from plato.serialization.safetensor import serialize_tree, deserialize_tree
import plato.utils.tree as tree
payload=[{"theta":torch.tensor([1.0])},{"version":1,"round":2,"client_id":1,"alpha":0.1,"run_id":"run","token":"dispatch"}]
recovered=deserialize_tree(serialize_tree(payload))
print(json.dumps({"source":tree.__file__,"types":{k:type(v).__name__ for k,v in recovered[1].items()},"values":{k:v.item() if isinstance(v,np.ndarray) and v.ndim==0 else v for k,v in recovered[1].items()},"tensor_restored":torch.equal(recovered[0]["theta"],payload[0]["theta"])}))
