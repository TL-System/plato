"""
A federated learning training session using the honest-but-curious server with gradient leakage attack.

Reference:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf
"""
import dlg_server


def main():
    """ A Plato federated learning training session using the honest-but-curious server with gradient leakage attack. """
    server = dlg_server.Server()
    server.run()


if __name__ == "__main__":
    main()
