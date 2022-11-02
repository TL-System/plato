

### Running qfed
./run -c examples/qfed/qfed_MNIST_lenet5.yml --cpu -b ./my_test

### Changed Files
/plato/processors/model_compress_qsgd.py

/plato/processors/model_decompress_qsgd.py

/plato/processors/registry.py

### Selected Log
[INFO][22:58:12]: [Client #2] Epoch: [5/5][620/625]     Loss: 0.000109

[INFO][22:58:12]: [Client #2] Model saved to ./my_test/models/pretrained/lenet5_2_72106.pth.

[INFO][22:58:12]: [Client #2] Loading a model from ./my_test/models/pretrained/lenet5_2_72106.pth.

[INFO][22:58:12]: [Client #2] Model trained.

[INFO][22:58:12]: [Client #2] Inbound data has been processed.

[INFO][22:58:12]: [Client #2] Outbound data is ready to be sent after being processed.

[INFO][22:58:12]: [Client #2] Global unstructured pruning applied.

parameter size of original, float16 and qsgd (in bytes):  456 447 51

parameter size of original, float16 and qsgd (in bytes):  1046 749 201

parameter size of original, float16 and qsgd (in bytes):  496 467 61

parameter size of original, float16 and qsgd (in bytes):  10047 5250 2451

parameter size of original, float16 and qsgd (in bytes):  912 675 165

parameter size of original, float16 and qsgd (in bytes):  192457 96460 48051

parameter size of original, float16 and qsgd (in bytes):  768 603 129

parameter size of original, float16 and qsgd (in bytes):  40757 20600 10127

parameter size of original, float16 and qsgd (in bytes):  472 455 55

parameter size of original, float16 and qsgd (in bytes):  3797 2120 887

[INFO][22:58:13]: Compression level: 1, **Original payload data size is 0.24 MB, quantized payload data size is 0.06 MB, compressed payload data size is 0.04 MB (simulated).**

[INFO][22:58:13]: [Client #2] Compressed model parameters.

[INFO][22:58:13]: [Client #2] Sent 0.04 MB of payload data to the server (simulated).

[INFO][22:58:13]: [Server #72102] Received 0.04 MB of payload data from client #2 (simulated).

[INFO][22:58:13]: [Server #72102] Decompressed received model parameters.

[INFO][22:58:13]: [Server #72102] Advancing the wall clock time to 1667357851.43.

[INFO][22:58:13]: [Server #72102] All 2 client report(s) received. Processing.

[INFO][22:58:13]: [Server #72102] Updated weights have been received.

[INFO][22:58:13]: [Server #72102] Aggregating model weight deltas.

[INFO][22:58:13]: [Server #72102] Finished aggregating updated weights.

[INFO][22:58:13]: [Server #72102] Started model testing.

[INFO][22:58:18]: [Server #72102] **Global model accuracy: 98.14%**

[INFO][22:58:18]: [Server #72102] All client reports have been processed.

[INFO][22:58:18]: [Server #72102] Saving the checkpoint to ./my_test/checkpoints/checkpoint_lenet5_1.pth.

[INFO][22:58:18]: [Server #72102] Model saved to ./my_test/checkpoints/checkpoint_lenet5_1.pth.

[INFO][22:58:18]: [Server #72102] Target accuracy reached.

[INFO][22:58:18]: [Server #72102] Training concluded.

[INFO][22:58:18]: [Server #72102] Model saved to ./my_test/models/pretrained/lenet5.pth.

[INFO][22:58:18]: [Server #72102] Closing the server.

[INFO][22:58:18]: Closing the connection to client #1.

[INFO][22:58:18]: [Client #1] The server disconnected the connection.  
