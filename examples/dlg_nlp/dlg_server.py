"""
An honest-but-curious federated learning server which can
analyze periodic gradients from certain clients to
perform the gradient leakage attacks and
reconstruct the training data of the victim clients.


References:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?"
in Advances in Neural Information Processing Systems 2020.

https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf
"""
import asyncio
import logging
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg
from torchvision import transforms

from utils.lamp import (
    fix_special_tokens,
    get_closest_tokens,
    get_reconstruction_loss,
    get_aux_lm,
    get_loss,
    swap_tokens,
)
from utils.modules import PatchedModule
from utils.utils import cross_entropy_for_onehot
from utils.utils import total_variation as TV
from utils.consts import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    LogitsProcessor,
    BeamSearchScorer,
)

cross_entropy = torch.nn.CrossEntropyLoss(reduce="mean")
tt = transforms.ToPILImage()

partition_size = Config().data.partition_size
epochs = Config().trainer.epochs
batch_size = Config().trainer.batch_size
num_iters = Config().algorithm.num_iters
log_interval = Config().algorithm.log_interval
dlg_result_path = f"{Config().params['result_path']}/{os.getpid()}"


class Server(fedavg.Server):
    """An honest-but-curious federated learning server with gradient leakage attack."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.attack_method = None
        self.share_gradients = True
        if (
            hasattr(Config().algorithm, "share_gradients")
            and not Config().algorithm.share_gradients
        ):
            self.share_gradients = False
        self.match_weights = False
        if (
            hasattr(Config().algorithm, "match_weights")
            and Config().algorithm.match_weights
        ):
            self.match_weights = True
        self.use_updates = True
        if (
            hasattr(Config().algorithm, "use_updates")
            and not Config().algorithm.use_updates
        ):
            self.use_updates = False
        # Save trail 1 as the best as default when results are all bad
        self.best_trial = 1

    def weights_received(self, weights_received):
        """
        Perform attack in attack around after the updated weights have been aggregated.
        """
        weights_received = [payload[0] for payload in weights_received]
        if (
            self.current_round == Config().algorithm.attack_round
            and Config().algorithm.attack_method in ["DLG", "iDLG", "csDLG"]
        ):
            self.attack_method = Config().algorithm.attack_method
            self._deep_leakage_from_gradients(weights_received)

        return weights_received

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging with optional compensation."""
        # Extract the total number of samples
        self.total_samples = sum([update.report.num_samples for update in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        _scale = 0
        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def _deep_leakage_from_gradients(self, weights_received):
        """Analyze periodic gradients from certain clients."""
        # Process data from the victim client
        # The ground truth should be used only for evaluation
        baseline_weights = self.algorithm.extract_weights()
        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received
        )
        update = self.updates[Config().algorithm.victim_client]
        target_weights = update.payload[0]
        if not self.share_gradients and self.match_weights and self.use_updates:
            target_weights = deltas_received[Config().algorithm.victim_client]

        gt_data, gt_labels, target_grad = (
            update.payload[1],
            update.payload[2],
            update.payload[3],
        )

        # Assume the reconstructed data shape is known, which can be also derived from the target dataset
        num_images = partition_size
        data_size = [num_images, gt_data.shape[1], gt_data.shape[2], gt_data.shape[3]]

        # The number of restarts
        trials = 1
        if hasattr(Config().algorithm, "trials"):
            trials = Config().algorithm.trials

        logging.info("Running %d Trials", trials)

        if not self.share_gradients and not self.match_weights:
            # Obtain the local updates from clients
            target_grad = []
            for delta in deltas_received[Config().algorithm.victim_client].values():
                target_grad.append(-delta / Config().parameters.optimizer.lr)

            total_local_steps = epochs * math.ceil(partition_size / batch_size)
            target_grad = [x / total_local_steps for x in target_grad]

        # Generate dummy items and initialize optimizer
        torch.manual_seed(Config().algorithm.random_seed)

        for trial_number in range(trials):
            self.run_trial(
                trial_number,
                num_images,
                data_size,
                target_weights,
                target_grad,
                gt_data,
                gt_labels,
            )

        self._save_best()

    def run_trial(
        self,
        trial_number,
        num_images,
        data_size,
        target_weights,
        target_grad,
        true_embeds,
        true_labels,
    ):
        """Run the attack for one trial."""
        logging.info("Starting Attack Number %d", (trial_number + 1))

        tokenizer = self.get_tokenizer()
        lm = get_aux_lm(Config().device())

        # BERT special tokens (0-999) are never part of the sentence
        unused_tokens = []
        if Config().algorithm.use_embedding:
            for i in range(tokenizer.vocab_size):
                if target_grad[0][i].abs().sum() < 1e-9 and i != BERT_PAD_TOKEN:
                    unused_tokens += [i]
        else:
            unused_tokens += list(range(1, 100))
            unused_tokens += list(range(104, 999))
        unused_tokens = np.array(unused_tokens)

        embeddings = self.trainer.model.get_input_embeddings()
        embeddings_weight = embeddings.weight.unsqueeze(0)

        pads = None

        # TODO: Target model updates
        x_embeds = self.get_init_embeds(
            unused_tokens,
            true_embeds.shape,
            true_labels,
            target_grad,
            embeddings,
            embeddings_weight,
            tokenizer,
            lm,
            pads,
        )

        # Init reconstruction optimizer
        if Config().algorithm.rec_optim == "Adam":
            match_optimizer = torch.optim.Adam([x_embeds], lr=Config().algorithm.rec_lr)
        elif Config().algorithm.rec_optim == "SGD":
            match_optimizer = torch.optim.SGD(
                [x_embeds], lr=0.01, momentum=0.9, nesterov=True
            )
        elif Config().algorithm.rec_optim == "LBFGS":
            match_optimizer = torch.optim.LBFGS(
                [x_embeds], lr=Config().algorithm.rec_lr
            )
        elif Config().algorithm.rec_optim == "bert-adam":
            match_optimizer = torch.optim.AdamW(
                [x_embeds],
                lr=Config().algorithm.rec_lr,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=0.01,
            )

        # Init learning rate scheduler
        if Config().algorithm.lr_decay_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.MStepLR(
                match_optimizer, step_size=50, gamma=Config().algorithm.lr_decay
            )
        elif Config().algorithm.lr_decay_type == "LambdaLR":

            def lr_lambda(current_step: int):
                return max(
                    0.0,
                    float(Config().algorithm.lr_max_it - current_step)
                    / float(max(1, Config().algorithm.lr_max_it)),
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(match_optimizer, lr_lambda)

        if pads is None:
            max_len = [x_embeds.shape[1]] * x_embeds.shape[0]
        else:
            max_len = pads

        # Conduct gradients/weights/updates matching
        if not self.share_gradients and self.match_weights:
            model = deepcopy(self.trainer.model.to(Config().device()))
            closure = self._weight_closure(
                match_optimizer, x_embeds, true_labels, target_weights, model
            )
        else:
            closure = self._gradient_closure(
                match_optimizer, x_embeds, true_labels, target_grad
            )

        best_final_error, best_final_x = None, x_embeds.detach().clone()
        for iters in range(num_iters):
            t_start = time.time()

            error = match_optimizer.step(closure)
            logging.info("Current error %f in iteration %d:", error, iters)

            if best_final_error is None or error <= best_final_error:
                best_final_error = error.item()
                best_final_x.data[:] = x_embeds.data[:]
            del error

            scheduler.step()

            fix_special_tokens(x_embeds, embeddings.weight, pads)

            _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, embeddings_weight)

            # Trying swaps
            if (
                Config().algorithm.use_swaps
                and iters
                >= Config().algorithm.swap_burnin * Config().algorithm.num_iters
                and iters % Config().algorithm.swap_every == 1
            ):
                swap_tokens(
                    x_embeds,
                    max_len,
                    cos_ids,
                    lm,
                    self.trainer.model,
                    true_labels,
                    target_grad,
                )

            steps_done = iters + 1
            if steps_done % Config().algorithm.print_every == 0:
                _, cos_ids = get_closest_tokens(
                    x_embeds, unused_tokens, embeddings_weight
                )
                x_embeds_proj = (
                    embeddings(cos_ids)
                    * x_embeds.norm(dim=2, p=2, keepdim=True)
                    / embeddings(cos_ids).norm(dim=2, p=2, keepdim=True)
                )
                _, _, tot_loss_proj = get_loss(
                    lm,
                    self.trainer.model,
                    cos_ids,
                    x_embeds_proj,
                    true_labels,
                    target_grad,
                )
                perplexity, rec_loss, tot_loss = get_loss(
                    lm, self.trainer.model, cos_ids, x_embeds, true_labels, target_grad
                )

                step_time = time.time() - t_start

                print(
                    "[%4d/%4d] tot_loss=%.3f (perp=%.3f, rec=%.3f), tot_loss_proj:%.3f [t=%.2fs]"
                    % (
                        steps_done,
                        Config().algorithm.num_iters,
                        tot_loss.item(),
                        perplexity.item(),
                        rec_loss.item(),
                        tot_loss_proj.item(),
                        step_time,
                    ),
                    flush=True,
                )
                print("prediction: %s" % (tokenizer.batch_decode(cos_ids)), flush=True)

                tokenizer.batch_decode(cos_ids)

    def _gradient_closure(self, match_optimizer, dummy_data, labels, target_grad):
        """Take a step to match the gradients."""

        def closure():
            match_optimizer.zero_grad()
            self.trainer.model.to(Config().device())
            # self.trainer.model.eval()
            """Should reconstruction be conducted in train() or eval() mode?"""
            self.trainer.model.zero_grad()
            try:
                dummy_pred, _ = self.trainer.model(dummy_data)
            except:
                dummy_pred = self.trainer.model(dummy_data)

            if self.attack_method == "DLG":
                dummy_onehot_label = F.softmax(labels, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
            elif self.attack_method in ["iDLG", "csDLG"]:
                dummy_loss = cross_entropy(dummy_pred, torch.argmax(labels, dim=-1))

            dummy_grad = torch.autograd.grad(
                dummy_loss, self.trainer.model.parameters(), create_graph=True
            )

            rec_loss = self._reconstruction_costs([dummy_grad], target_grad)
            if (
                hasattr(Config().algorithm, "total_variation")
                and Config().algorithm.total_variation > 0
            ):
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            if self.attack_method == "csDLG":
                dummy_data.grad.sign_()
            return rec_loss

        return closure

    def _weight_closure(
        self, match_optimizer, dummy_data, labels, target_weights, model
    ):
        """Take a step to match the weights."""

        def closure():
            match_optimizer.zero_grad()
            dummy_weight = self._loss_steps(dummy_data, labels, model)

            rec_loss = self._reconstruction_costs(
                [dummy_weight], list(target_weights.values())
            )
            if (
                hasattr(Config().algorithm, "total_variation")
                and Config().algorithm.total_variation > 0
            ):
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def _loss_steps(self, dummy_data, labels, model):
        """Take a few gradient descent steps to fit the model to the given input."""
        patched_model = PatchedModule(model)
        if self.use_updates:
            patched_model_origin = deepcopy(patched_model)

        for epoch in range(epochs):
            if batch_size == 1:
                dummy_pred = patched_model(dummy_data, patched_model.parameters)
                labels_ = labels
            else:
                idx = epoch % (dummy_data.shape[0] // batch_size)
                dummy_pred = patched_model(
                    dummy_data[idx * batch_size : (idx + 1) * batch_size],
                    patched_model.parameters,
                )
                labels_ = labels[idx * batch_size : (idx + 1) * batch_size]

            loss = cross_entropy(dummy_pred, labels_).sum()

            grad = torch.autograd.grad(
                loss,
                patched_model.parameters.values(),
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )

            patched_model.parameters = OrderedDict(
                (name, param - Config().parameters.optimizer.lr * grad_part)
                for ((name, param), grad_part) in zip(
                    patched_model.parameters.items(), grad
                )
            )
        if self.use_updates:
            patched_model.parameters = OrderedDict(
                (name, param - param_origin)
                for ((name, param), (name_origin, param_origin)) in zip(
                    patched_model.parameters.items(),
                    patched_model_origin.parameters.items(),
                )
            )
        return list(patched_model.parameters.values())

    @staticmethod
    def _reconstruction_costs(dummy, target):
        indices = torch.arange(len(target))

        ex = target[0]
        if Config().algorithm.cost_weights == "linear":
            weights = torch.arange(
                len(target), 0, -1, dtype=ex.dtype, device=ex.device
            ) / len(target)
        elif Config().algorithm.cost_weights == "exp":
            weights = torch.arange(len(target), 0, -1, dtype=ex.dtype, device=ex.device)
            weights = weights.softmax(dim=0)
            weights = weights / weights[0]
        else:
            weights = target[0].new_ones(len(target))

        cost_fn = Config().algorithm.cost_fn

        total_costs = 0
        for trial in dummy:
            pnorm = [0, 0]
            costs = 0
            for i in indices:
                if cost_fn == "l2":
                    costs += ((trial[i] - target[i]).pow(2)).sum() * weights[i]
                elif cost_fn == "l1":
                    costs += ((trial[i] - target[i]).abs()).sum() * weights[i]
                elif cost_fn == "max":
                    costs += ((trial[i] - target[i]).abs()).max() * weights[i]
                elif cost_fn == "sim":
                    costs -= (trial[i] * target[i]).sum() * weights[i]
                    pnorm[0] += trial[i].pow(2).sum() * weights[i]
                    pnorm[1] += target[i].pow(2).sum() * weights[i]
                elif cost_fn == "simlocal":
                    costs += (
                        1
                        - torch.nn.functional.cosine_similarity(
                            trial[i].flatten(), target[i].flatten(), 0, 1e-10
                        )
                        * weights[i]
                    )
            if cost_fn == "sim":
                costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

            # Accumulate final costs
            total_costs += costs

        return total_costs / len(dummy)

    @staticmethod
    def get_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        tokenizer.model_max_length = 512

        return tokenizer

    def get_init_embeds(
        self,
        unused_tokens,
        shape,
        true_labels,
        target_grad,
        bert_embeddings,
        bert_embeddings_weight,
        tokenizer,
        lm,
        pads,
    ):
        device = lm.device
        num_inits = shape[0]
        lm_tokenizer = tokenizer

        # Generate candidates from language model / random
        if Config().algorithm.init == "lm":
            sentence = "the"
            input_ids = lm_tokenizer.encode(sentence, return_tensors="pt").to(device)[
                :, 1:-1
            ]
            init_len = 10
            gen_outs = lm.generate(
                input_ids,
                no_repeat_ngram_size=2,
                num_return_sequences=Config().algorithm.init_candidates * num_inits,
                do_sample=True,
                max_length=shape[1] + init_len,
            )
            gen_outs = gen_outs[:, init_len:]
            all_candidates = lm_tokenizer.batch_decode(gen_outs)
            embeds = tokenizer(
                all_candidates, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"].to(device)
            embeds = bert_embeddings(embeds)[:, : shape[1], :]
        elif Config().algorithm.init == "random":
            new_shape = [Config().algorithm.init_candidates * num_inits] + list(
                shape[1:]
            )
            embeds = torch.randn(new_shape).to(device)

        # Pick candidates based on rec loss
        best_x_embeds, best_rec_loss = None, None
        for i in range(Config().algorithm.init_candidates):
            tmp_embeds = embeds[i * num_inits : (i + 1) * num_inits]
            fix_special_tokens(tmp_embeds, bert_embeddings.weight, pads)

            rec_loss = get_reconstruction_loss(
                self.trainer.model, tmp_embeds, true_labels, target_grad
            )
            if (best_rec_loss is None) or (rec_loss < best_rec_loss):
                best_rec_loss = rec_loss
                best_x_embeds = tmp_embeds
                _, cos_ids = get_closest_tokens(
                    tmp_embeds, unused_tokens, bert_embeddings_weight, metric="cos"
                )
                sen = tokenizer.batch_decode(cos_ids)
                print(
                    f"[Init] best rec loss: {best_rec_loss.item()} for {sen}",
                    flush=True,
                )

        # Pick best permutation of candidates
        for i in range(Config().algorithm.init_candidates):
            idx = torch.cat(
                (
                    torch.tensor([0], dtype=torch.int32),
                    torch.randperm(shape[1] - 2) + 1,
                    torch.tensor([shape[1] - 1], dtype=torch.int32),
                )
            )
            tmp_embeds = best_x_embeds[:, idx].detach()
            rec_loss = get_reconstruction_loss(
                self.trainer.model, tmp_embeds, true_labels, target_grad
            )
            if rec_loss < best_rec_loss:
                best_rec_loss = rec_loss
                best_x_embeds = tmp_embeds
                _, cos_ids = get_closest_tokens(
                    tmp_embeds, unused_tokens, bert_embeddings_weight, metric="cos"
                )
                sen = tokenizer.batch_decode(cos_ids)
                print(
                    f"[Init] best perm rec loss: {best_rec_loss.item()} for {sen}",
                    flush=True,
                )

        # Scale inital embeddings to Config().algorithm.init_size (e.g., avg of BERT embeddings ~1.4)
        if Config().algorithm.init_size >= 0:
            best_x_embeds /= best_x_embeds.norm(dim=2, keepdim=True)
            best_x_embeds *= Config().algorithm.init_size

        x_embeds = best_x_embeds.detach().clone()
        x_embeds = x_embeds.requires_grad_(True)

        return x_embeds
