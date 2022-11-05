
# qfed
The first version of qfed only achieved the quantization and compression based on qsgd with some modification. On the simplest experiment I have conducted now, it looks good. See more details in the [README](https://github.com/TL-System/plato/blob/QFed/examples/qfed/README.md) and [setting](https://github.com/TL-System/plato/blob/QFed/examples/qfed/qfed_MNIST_lenet5.yml) files.

### Running qfed
./run -c examples/qfed/qfed_MNIST_lenet5.yml --cpu -b ./my_test

### Changed Files
/plato/processors/model_compress_qsgd.py

/plato/processors/model_decompress_qsgd.py

/plato/processors/registry.py