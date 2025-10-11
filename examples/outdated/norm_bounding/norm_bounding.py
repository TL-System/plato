"""
A FedAvg training session using norm bounding defense

Reference:
Sun, Z., Kairouz, P., Suresh, A. T., & McMahan, H. B. (2019). Can you really backdoor
federated learning?. arXiv preprint arXiv:1911.07963.

https://arxiv.org/pdf/1911.07963.pdf
"""

import norm_bounding_server


def main():
    server = norm_bounding_server.Server()
    server.run()


if __name__ == "__main__":
    main()
