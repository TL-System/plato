"""
A federated learning training session using FedTP.

A novel Transformer-based federated learning framework with personalized self-attention
to better handle data heterogeneity among clients. 
FedTP learns a personalized self-attention layer for each client 
while the parameters of the other layers are shared among the clients.

Reference:
Li, Hongxia, Zhongyi Cai, Jingya Wang, Jiangnan Tang, Weiping Ding, Chin-Teng Lin, and Ye Shi.
"FedTP: Federated Learning by Transformer Personalization."
arXiv preprint arXiv:2211.01572 (2022).

https://arxiv.org/pdf/2211.01572v1.pdf.
"""
import fedtp_server
import vit

from plato.clients.simple import Client
from plato.config import Config


def main():
    """A Plato federated learning training session using FedProx."""
    if Config().trainer.model_name == "FedTP_ViT":
        model = vit.ViT
    else:
        model = None

    client = Client(model=model)
    server = fedtp_server.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
