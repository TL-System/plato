"""
Starting point for a Plato federated learning training session.
"""

import os

from plato.servers import registry as server_registry

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    server = server_registry.get()
    server.run()


if __name__ == "__main__":
    main()
