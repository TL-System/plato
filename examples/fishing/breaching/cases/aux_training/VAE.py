"""Train autoencoders."""

import torch
import torch.nn.functional as F

from .nearest_embed import NearestEmbed


class AE(torch.nn.Module):
    """Basic deterministic auto-encoder."""

    def __init__(self, feature_model, decoder):
        super().__init__()

        self.encoder = feature_model
        self.decoder = decoder

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code), None

    def loss(self, x, recon_x, *args, data_mean=0.0, data_std=1.0):
        """Based on https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py#L93.
        Compare BCE on unnormalized images."""
        mse = F.mse_loss(recon_x * data_std, x * data_std)
        return mse


class VAE(torch.nn.Module):
    """Closely following https://github.com/pytorch/examples/blob/master/vae/main.py."""

    def __init__(self, feature_model, decoder, kl_coef=1.0):
        super().__init__()

        self.encoder = feature_model
        self.decoder = decoder

        self.kl_coef = kl_coef

    def reparameterize(self, mu, logvar, noise_level=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_level
        return mu + eps * std

    def forward(self, x, noise_level=1.0):
        code = self.encoder(x).flatten(start_dim=1)
        cutoff = code.shape[1] // 2
        mu, logvar = code[:, :cutoff], code[:, cutoff:]
        z = self.reparameterize(mu, logvar, noise_level)
        return self.decoder(torch.cat([z] * 2, dim=1)), mu, logvar

    def loss(self, x, recon_x, mu, logvar, data_mean=0.0, data_std=1.0):
        """Based on https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py#L93.
        Compare BCE on unnormalized images."""
        B = x.shape[0]
        bce = F.binary_cross_entropy(recon_x * data_std + data_mean, x * data_std + data_mean, reduction="sum")

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + self.kl_coef * kl


class VQ_VAE(torch.nn.Module):
    """Vector Quantized AutoEncoder from https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py#L110"""

    def __init__(self, feature_model, decoder, k=512, vq_coef=0.2, mse_coef=0.4, **kwargs):
        super().__init__()

        self.emb_size = k
        self.encoder = feature_model
        self.decoder = decoder

        self.embedding = NearestEmbed(k, k)

        self.vq_coef = vq_coef
        self.mse_coef = mse_coef

    def forward(self, x):
        code = self.encoder(x)
        z_e = code.view(code.shape[0], self.emb_size, -1)
        z_q, _ = self.embedding(z_e, weight_sg=True)
        emb, _ = self.embedding(z_e.detach())
        return self.decoder(z_q), z_e.view_as(code), emb.view_as(code)

    def loss(self, x, recon_x, z_e, emb, data_mean=0.0, data_std=1.0):
        B = x.shape[0]
        bce_loss = F.binary_cross_entropy(recon_x * data_std + data_mean, x * data_std + data_mean, reduction="sum")
        vq_loss = F.mse_loss(emb, z_e.detach())
        mse_loss = F.mse_loss(z_e, emb.detach())

        return bce_loss + self.vq_coef * vq_loss + self.mse_coef * mse_loss


class VQ_CVAE(torch.nn.Module):
    def __init__(self, feature_model, decoder, d=512, k=10, vq_coef=1, commit_coef=0.5, **kwargs):
        super().__init__()

        self.encoder = feature_model
        self.decoder = decoder

        self.d = d
        self.embedding = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef

        with torch.no_grad():
            self.embedding.weight.normal_(0, 0.02)
            torch.fmod(self.embedding.weight, 0.04)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, argmin = self.embedding(z_e, weight_sg=True)
        emb, _ = self.embedding(z_e.detach())
        return self.decoder(z_q), z_e, emb, argmin

    def loss(self, x, recon_x, z_e, emb, argmin, data_mean=0.0, data_std=1.0):
        mse = F.mse_loss(recon_x + data_std, x * data_std)

        vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        commit_loss = torch.mean(torch.norm((emb.detach() - z_e) ** 2, 2, 1))
        return mse + self.vq_coef * vq_loss + self.commit_coef * commit_loss


def train_encoder_decoder(encoder, decoder, dataloader, setup, arch="AE"):
    """Train a VAE."""
    epochs = 250
    lr = 1e-2
    data_mean = torch.as_tensor(dataloader.dataset.mean, **setup)[None, :, None, None]
    data_std = torch.as_tensor(dataloader.dataset.std, **setup)[None, :, None, None]

    if arch == "AE":
        model = AE(encoder, decoder)
    elif arch == "VAE":
        model = VAE(encoder, decoder, kl_coef=1.0)
    elif arch == "VQ_VAE":
        model = VQ_VAE(encoder, decoder)
    elif arch == "VQ_CVAE":
        model = VQ_CVAE(encoder, decoder)
    else:
        raise ValueError("Invalid model.")
    model.to(**setup)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0)
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_mse, epoch_test = 0, 0, 0
        for idx, (data, label) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            data = data.to(**setup)

            reconstructed_data, *internal_vars = model(data)
            loss = model.loss(data, reconstructed_data, *internal_vars, data_mean=data_mean, data_std=data_std)
            loss.backward()
            with torch.inference_mode():
                model.eval()
                # epoch_test += F.mse_loss(data, model(data, noise_level=0.0)[0])
                epoch_loss += loss.detach()
                epoch_mse += F.mse_loss(data, reconstructed_data)
                optimizer.step()

        print(f"Epoch {epoch}_{idx}: Avg. Loss: {epoch_loss / (idx + 1)}. Avg. MSE: {epoch_mse / (idx + 1)}")
        # print(f'Epoch {epoch}: Avg. Loss: {epoch_loss / (idx + 1)}. Avg. MSE: {epoch_mse / (idx + 1)}. Avg. Test: {epoch_test / (idx + 1)}')
    model.eval()


def status_message(optimizer, stats, step):
    """A basic console printout."""
    current_lr = f'{optimizer.param_groups[0]["lr"]:.4f}'

    def _maybe_print(key):
        return stats[key][-1] if len(stats[key]) > 0 else float("NaN")

    msg = f'Step: {step:<4}| lr: {current_lr} | Time: {stats["train_time"][-1]:4.2f}s |'
    msg += f'TRAIN loss {stats["train_loss"][-1]:7.4f} | TRAIN Acc: {stats["train_acc"][-1]:7.2%} |'
    msg += f'VAL loss {_maybe_print("valid_loss"):7.4f} | VAL Acc: {_maybe_print("valid_acc"):7.2%} |'
    return msg
