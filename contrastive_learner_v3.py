import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from tqdm import tqdm
import torch

import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import dataloaders.patch_triplet_dataloaders as patch3_dloaders
import dataloaders.util_dataloaders
import dataloaders.util_dataloaders as util_dloaders
import os
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
import constants
from miners import SimpleMiner, SimpleEMAMiner, FullEMAMiner, SimpleRandomMiner

from utils.torch_unified_logger import UnifiedLogger, to_wandb_config
from utils.gradnorm import GradNorm
from pytorch_metric_learning import losses, distances

from transforms import (
    GaussianNoiseTransform,
    RandomRotationTransform,
    RandomTranslationTransform,
    RandomCutTransform
)

# SEMIHARD_FACTOR = 4

TRIPLET_MINING_SEMIHARD = 'semihard'
TRIPLET_MINING_MOCO = 'moco'
TRIPLET_MINING_NONE = 'none'

from utils.debug import in_debugger, is_emb_tensor_normalized
from feature_extractors.encoders_base import instantiate_model
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

class EarlyStopping:
    """ A custom, bastardish implementation of early stopped for our net"""

    def __init__(self, patience=5, min_delta_loss=0, min_delta_margin=0):
        self.patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_margin = min_delta_margin
        self.counter = 0
        self.best_loss = None
        self.best_margin = None
        # self.early_stop = False

    def __call__(self, val_loss, distance_margin):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta_loss:
            self.best_loss = val_loss
            self.counter = 0
            # self.early_stop = False
            return False
        if self.best_margin is None or distance_margin > self.best_margin + self.min_delta_margin:
            self.best_margin = distance_margin
            self.counter = 0
            # self.early_stop = False
            return False
        self.counter += 1
        if self.counter >= self.patience:
            # self.early_stop = True
            return True


class MarginSchedulerFixed:
    def __init__(self, margin):
        self.base_margin = margin
        self.hard_yield = 0.0
        self.semihard_yield = 0.0
        self.easy_yield = 0.0

    def update_margin(self, current_margin):
        return current_margin, False

    def _assert_sum_yields(self):
        sum_yields = self.hard_yield + self.semihard_yield + self.easy_yield
        assert (np.isclose(sum_yields, 1, atol=1e-3)), \
            f'Sum of yields is {sum_yields}'

    def update_yields(self, hard_yield, semihard_yield, easy_yield):
        self.hard_yield = hard_yield
        self.semihard_yield = semihard_yield
        self.easy_yield = easy_yield
        self._assert_sum_yields()


class MarginSchedulerOnYield(MarginSchedulerFixed):
    def __init__(self, margin, min_margin_mult=0.9, max_margin_mult=1.5):
        super().__init__(margin)
        self.min_margin_mult = min_margin_mult
        self.max_margin_mult = max_margin_mult
        self.hard_yield = 0.0
        self.semihard_yield = 0.0
        self.easy_yield = 0.0
        self.min_yield = 0.5
        self.max_yield = 0.75

    def update_margin(self, current_margin):
        self._assert_sum_yields()
        # Adjust miner margin depending on the yield
        if (self.hard_yield + self.semihard_yield) < self.min_yield:
            new_margin = current_margin * 1.1
            if new_margin > self.base_margin * self.max_margin_mult:
                return self.base_margin * self.max_margin_mult, True
            return new_margin, True
        if (self.hard_yield + self.semihard_yield) > self.max_yield:
            new_margin = current_margin * 0.9
            if new_margin < self.base_margin * self.min_margin_mult:
                return self.base_margin * self.min_margin_mult, True
            return new_margin, True
        return current_margin, False

@dataclass
class BatchStats:
    # Statistics for fully-aligned tuples (sharing the same center across shapes)
    sims_fa_ap: List[np.ndarray]
    sims_fa_an: List[np.ndarray]
    margins_fa: List[np.ndarray]
    # Mined, or weakly-aligned tuples
    sims_mined_ap: List[np.ndarray]
    sims_mined_an: List[np.ndarray]
    margins_mined: List[np.ndarray]

    def __init__(self):
        self.clear()

    def clear(self):
        self.sims_fa_ap = []
        self.sims_fa_an = []
        self.margins_fa = []
        self.sims_mined_ap = []
        self.sims_mined_an = []
        self.margins_mined = []

    def update_1forward(self, emb_all, mined_indices_tuple, fa_indices_tuple):
        sim_mined_ap, sim_mined_an, margin_mined = compute_fa_similarities(emb_all, mined_indices_tuple)
        self.sims_mined_ap.append(sim_mined_ap.detach().cpu().numpy())
        self.sims_mined_an.append(sim_mined_an.detach().cpu().numpy())
        self.margins_mined.append(margin_mined.detach().cpu().numpy())

        sim_fa_ap, sim_fa_an, margin_fa = compute_fa_similarities(emb_all, fa_indices_tuple)
        self.sims_fa_ap.append(sim_fa_ap.detach().cpu().numpy())
        self.sims_fa_an.append(sim_fa_an.detach().cpu().numpy())
        self.margins_fa.append(margin_fa.detach().cpu().numpy())

    def update_2forward(self, emb_mined, mined_indices_tuple, emb_fa, fa_indices_tuple):
        sim_mined_ap, sim_mined_an, margin_mined = compute_fa_similarities(emb_mined, mined_indices_tuple)
        self.sims_mined_ap.append(sim_mined_ap.detach().cpu().numpy())
        self.sims_mined_an.append(sim_mined_an.detach().cpu().numpy())
        self.margins_mined.append(margin_mined.detach().cpu().numpy())

        sim_fa_ap, sim_fa_an, margin_fa = compute_fa_similarities(emb_fa, fa_indices_tuple)
        self.sims_fa_ap.append(sim_fa_ap.detach().cpu().numpy())
        self.sims_fa_an.append(sim_fa_an.detach().cpu().numpy())
        self.margins_fa.append(margin_fa.detach().cpu().numpy())

    def aggregate_and_report(self, writer, global_step):
        sim_fa_ap = np.concatenate(self.sims_fa_ap, axis=0)
        sim_fa_an = np.concatenate(self.sims_fa_an, axis=0)
        margin_fa = np.concatenate(self.margins_fa, axis=0)
        sim_mined_ap = np.concatenate(self.sims_mined_ap, axis=0)
        sim_mined_an = np.concatenate(self.sims_mined_an, axis=0)
        margin_mined = np.concatenate(self.margins_mined, axis=0)

        # Changed from torch.mean to np.mean
        mean_sim_fa_ap = np.mean(sim_fa_ap)
        mean_sim_fa_an = np.mean(sim_fa_an)
        mean_margin_fa = np.mean(margin_fa)
        mean_sim_mined_ap = np.mean(sim_mined_ap)
        mean_sim_mined_an = np.mean(sim_mined_an)
        mean_margin_mined = np.mean(margin_mined)

        writer.log_scalars('Similarity/FASimilarity', {
            'Positive': mean_sim_fa_ap,
            'Negative': mean_sim_fa_an,
            'Margin': mean_margin_fa
        }, global_step, log_together=True)
        writer.log_scalars('Similarity/MinedSimilarity', {
            'Positive': mean_sim_mined_ap,
            'Negative': mean_sim_mined_an,
            'Margin': mean_margin_mined
        }, global_step, log_together=True)
        writer.log_histogram('Similarity/FAPositiveDistribution', np.array(sim_fa_ap), global_step)
        writer.log_histogram('Similarity/FANegativeDistribution', np.array(sim_fa_an), global_step)
        writer.log_histogram('Similarity/MinedPositiveDistribution', np.array(sim_mined_ap), global_step)
        writer.log_histogram('Similarity/MinedNegativeDistribution', np.array(sim_mined_an), global_step)
        return {
            'mean_sim_fa_ap': mean_sim_fa_ap,
            'mean_sim_fa_an': mean_sim_fa_an,
            'mean_margin_fa': mean_margin_fa,
            'mean_sim_mined_ap': mean_sim_mined_ap,
            'mean_sim_mined_an': mean_sim_mined_an,
            'mean_margin_mined': mean_margin_mined,
        }



def compute_fa_similarities(embeddings, indices_tuple):
    """
    Compute cosine similarities for all fully-aligned (sharing same coordinates) tuples

    Returns:
        s_ap: (K,) similarities for anchor-positive
        s_an: (K,) similarities for anchor-negative
        margins: (K,) margins (s_ap - s_an) - note reversed for similarity
    """

    a_pos, p_pos, a_neg, n_neg = indices_tuple
    # Find shared anchors
    shared_mask = a_pos.unsqueeze(1) == a_neg.unsqueeze(0)
    pos_idx, neg_idx = torch.where(shared_mask)

    if len(pos_idx) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Get indices
    anchors = a_pos[pos_idx]
    positives = p_pos[pos_idx]
    negatives = n_neg[neg_idx]

    # Get normalized embeddings (assuming already L2-normalized)
    anchor_emb = embeddings[anchors].detach().cpu()
    positive_emb = embeddings[positives].detach().cpu()
    negative_emb = embeddings[negatives].detach().cpu()

    # Compute cosine similarities
    s_ap = (anchor_emb * positive_emb).sum(dim=1)  # (K,)
    s_an = (anchor_emb * negative_emb).sum(dim=1)  # (K,)

    # For similarity, margin is reversed (we want s_ap > s_an)
    margins = s_ap - s_an  # (K,)

    assert s_ap.shape == margins.shape
    assert s_an.shape == margins.shape
    return s_ap.detach(), s_an, margins


class MSTrainer:
    """Simple trainer for contrastive learning"""

    def __init__(self, device, model, cfg, timestamp=None):
        self.model = model
        self.device = device
        # self.margin = cfg.training.triplet_loss_margin
        # self.batch_size = cfg.training.batch_size
        self.miner_type = cfg.triplet_mining.miner
        self.margin_scheduler = MarginSchedulerFixed(cfg.loss.triplet_loss_margin)

        self.reduction = 'none'
        self.current_margin = cfg.loss.triplet_loss_margin

        # Here we initialize the loss for Weakly Aligned tuples.
        if cfg.loss.wa_loss_type == 'MS':
            logging.warning('Initialized WA loss to MultiSimilarityLoss')
            self.ms_loss_mined =  losses.MultiSimilarityLoss(alpha=cfg.loss.mined_alpha, beta=cfg.loss.mined_beta,
                                                             base=self.current_margin)
        elif cfg.loss.wa_loss_type == 'SupCon':
            logging.warning('Initialized WA loss to SupConLoss')
            self.ms_loss_mined =  losses.SupConLoss(
                temperature=cfg.loss.wa_loss_temperature, distance=distances.CosineSimilarity())
        else:
            raise NotImplementedError(f'Unknown loss type {cfg.loss.wa_loss_type}')

        self.ms_loss_forced = losses.MultiSimilarityLoss(alpha=cfg.loss.forced_alpha, beta=cfg.loss.forced_beta,
                                                         base=self.current_margin)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                           weight_decay=cfg.training.weight_decay)

        # It may make sense to try different schedulers
        self.scheduler = self.create_training_scheduler(cfg)

        # Set up a simplest Dynamic Weighting with momentum
        self.use_grad_norm = cfg.loss.use_grad_norm
        self.fa_loss_weight = float(cfg.loss.fa_loss_weight)
        self.wa_loss_weight = float(cfg.loss.wa_loss_weight)
        model_dtype = next(model.parameters()).dtype
        self.loss_weighter = GradNorm(
            init_weights=torch.tensor(
                [self.wa_loss_weight, self.fa_loss_weight],
                dtype=model_dtype, device=self.device),
            model=model,
            num_losses=2,
            alpha=0.5,
            iters_to_check_grad_conflict=10)

        # Statistics collection
        self.epoch = 0
        self.global_step = 0

        # Determine the effective run ID for directory naming.
        # In a DVC pipeline run, cfg.experiment.run_id is set to pipeline.weights_run_id
        # (e.g. "20260318_142200"), grouping all splits under the same top-level directory.
        # In a manual run (non-DVC), run_id is null and we fall back to the per-invocation
        # timestamp so each manual run gets its own directory and nothing is overwritten.
        if not timestamp:
            timestamp = UnifiedLogger.get_timestamp()
        effective_run_id = str(cfg.experiment.run_id) if cfg.experiment.run_id else timestamp

        # Weight and log paths follow the layout:
        #   {model_output_path}/{effective_run_id}/{data.split}/
        # e.g. /data/model_weights/20260318_142200/train_classes_3/
        # DVC overrides model_output_path to /data/model_weights; for manual runs it
        # defaults to ./data/model_weights/manual (see configs/config_riconv_msloss_v3.yaml).
        self.weights_dir = os.path.join(cfg.paths.model_output_path, effective_run_id, cfg.data.split)
        self.log_dir = os.path.join(cfg.paths.logs_output_path, effective_run_id, cfg.data.split)
        os.makedirs(self.weights_dir, exist_ok=True)

        # Write a start marker immediately — visible while training is in progress.
        open(os.path.join(self.weights_dir, f'train_start_{effective_run_id}'), 'w').close()
        self.writer = UnifiedLogger(SummaryWriter(self.log_dir), use_wandb=cfg.experiment.use_wandb)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        print(f"To view logs, run: tensorboard --logdir {self.log_dir}")

        # Early stopping parameters
        self.best_val_loss = float('inf')
        self.best_distance_margin = -float('inf')
        self.patience_counter = 0

        self.single_batch = cfg.experiment.single_batch

    def set_miner(self, miner):
        self.miner = miner

    def create_training_scheduler(self, cfg):
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=15, cooldown=2, threshold=0.01)
        base_lr = cfg.training.lr
        if self.miner_type != TRIPLET_MINING_NONE:
            base_lr = base_lr * math.sqrt(1 / cfg.triplet_mining.crit_yield)
        min_lr = base_lr * 5e-2
        warmup_epochs = cfg.training.epochs / 20
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs - warmup_epochs, eta_min=min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return scheduler

    def update_distance_stats(self, anchor_emb, positive_emb, negative_emb, positive_distances, negative_distances):
        # Track distances for analysis
        pos_dist = torch.norm(anchor_emb - positive_emb, dim=1)
        neg_dist = torch.norm(anchor_emb - negative_emb, dim=1)
        positive_distances.extend(pos_dist.cpu().numpy())
        negative_distances.extend(neg_dist.cpu().numpy())
        return positive_distances, negative_distances


    def log_gradients(self, step):
        """Log model gradients to TensorBoard"""
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Log gradient norms for each layer
                param_norm = param.grad.data.norm(2).item()
                # total_norm += param_norm.item() ** 2
                self.writer.log_scalar(f'GradientsNorm/{name}', param_norm, step)

                # Log gradient histograms (less frequently to save space)
                if step % 100 == 0:
                    self.writer.log_histogram(f'GradientsHist/{name}', param.grad.data.cpu().numpy(), step)

    def save_checkpoint(self, filename, val_metrics=None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'val_metrics': val_metrics
        }
        filepath = os.path.join(self.weights_dir, filename)
        os.makedirs(self.weights_dir, exist_ok=True)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        return checkpoint.get('val_metrics', {})


    def batch_forward(self, batch_mined, batch_fa):
        """Run a single forward pass through the model with a batch of data.

        Args:
            batch_mined: mined patches, i.e. weakly-aligned patch tuples that don't share coordinates
            batch_fa: fully-alligned tuples that contain actual anomaly
        """
        Bm = batch_mined['patches'].shape[0]
        Bt = batch_fa['patches'].shape[0]

        # Augment the batch with mined negative samples
        (mined_patches, mined_indices, emb_mined_ema, z_mined_ema),(fa_patches, fa_indices, emb_fa_ema, _) = self.miner.mine(
            batch_mined['patches'], batch_mined['valid_indices'],
            batch_fa['patches'], batch_fa['valid_indices'])

        if self.single_batch:
            assert emb_mined_ema is None
            assert emb_fa_ema is None
            # Unite FA and WA tuples that come coming from two different sources (i.e. two sub-batches of data)
            # into a single batch passes them through the model as a single batch of data.
            # Collating should not be required for a dual pass.
            patches, mined_indices_concat, fa_indices_concat = patch3_dloaders.collate_ms_batches(
                mined_patches, mined_indices, fa_patches, fa_indices)
            patches = patches.to(self.device)
            mined_indices_concat = tuple(t.to(self.device) for t in mined_indices_concat)
            fa_indices_concat = tuple(t.to(self.device) for t in fa_indices_concat)

            emb_all, z_mined, z_fa = self.model(patches)
            emb_mined = emb_fa = None
        else:
            # Pass FA and WA sub-batches separately through the model. Accumulate gradients
            assert emb_fa_ema is None   # This is an invariant for now - we don't do any mining on FA tuples
            mined_patches = mined_patches.to(self.device)
            mined_indices_concat = tuple(t.to(self.device) for t in mined_indices)
            fa_patches = fa_patches.to(self.device)
            fa_indices_concat = tuple(t.to(self.device) for t in fa_indices)
            emb_mined_ema = emb_mined_ema.to(self.device)
            z_mined_ema = z_mined_ema.to(self.device)

            # Set these to none to avoid using them accidentally later
            fa_indices = mined_indices = None

            # Run forward passes.
            emb_mined, z_mined, _ = self.model(mined_patches)

            if emb_mined_ema is not None:
                if not is_emb_tensor_normalized(emb_mined_ema):
                    logger.critical(f'emb_mined_ema is not normalized')
                if not is_emb_tensor_normalized(z_mined_ema):
                    logger.critical(f'z_mined_ema is not normalized')

                # TODO(alexta): investigation / debug only. Here we add back patches from FA tuples.
                # sim_ema_vs_trainer = torch.sum(z_mined_ema * z_mined[Bm:], dim=1) # cosine similarity
                # avg_sim_ema_vs_trainer = sim_ema_vs_trainer.mean()
                # self.writer.log_scalar('EMADebug/embed_sim', avg_sim_ema_vs_trainer, self.global_step)

                emb_mined = torch.cat((emb_mined, emb_mined_ema), dim=0)
                z_mined = torch.cat((z_mined, z_mined_ema), dim=0)
                assert mined_indices_concat[0].max() < emb_mined.shape[0]
                assert mined_indices_concat[1].max() < emb_mined.shape[0]
                assert mined_indices_concat[2].max() < emb_mined.shape[0]
                assert mined_indices_concat[3].max() < emb_mined.shape[0]

            emb_fa, _, z_fa = self.model(fa_patches)
            emb_all = None

        if not is_emb_tensor_normalized(z_mined):
            logger.critical(f'z_mined_mean is not normalized')
        if not is_emb_tensor_normalized(z_fa):
            logger.critical(f'z_mined_mean is not normalized')
        if emb_all is not None and not is_emb_tensor_normalized(emb_all):
            logger.critical(f'emb_all is not normalized')

        loss_mined = self.ms_loss_mined(z_mined, labels=None, indices_tuple=mined_indices_concat)
        loss_fa = self.ms_loss_forced(z_fa, labels=None, indices_tuple=fa_indices_concat)

        return emb_all, emb_mined, emb_fa, loss_mined, loss_fa, mined_indices_concat, fa_indices_concat, Bm, Bt

    def train_epoch_fa(self, neg_dataloader):
        self.model.train()

        loss_fa = None
        total_forced_loss = 0

        num_batches = 0

        pbar = tqdm(enumerate(neg_dataloader), total=len(neg_dataloader))
        for _, batch_fa in pbar:
            patches = batch_fa['patches'].to(self.device)
            indices = tuple(t.to(self.device) for t in batch_fa['valid_indices'])
            _, _, z_fa = self.model(patches)
            loss_fa = self.ms_loss_forced(z_fa, labels=None, indices_tuple=indices)

            self.optimizer.zero_grad()
            # Backward pass
            loss_fa.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Do param optimization - a single step
            self.optimizer.step()

            total_forced_loss += loss_fa.item()
            num_batches += 1
            # Add the number of fully aligned tuples we've seen so far as a global step
            self.global_step += patches.shape[0]
        total_forced_loss /= num_batches
        self.writer.log_scalar('ForcedLoss/Training', loss_fa, self.global_step)
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.log_scalar('LearningRate/Current', current_lr, self.global_step)

        self.epoch += 1
        return {
            'train_sum_loss': total_forced_loss,
            'train_mined_loss': 0,
            'train_forced_loss': total_forced_loss
        }



    def train_epoch_wa(self, all_pos_dataloader, neg_dataloader):
        """Ablation: trains a single epoch on WA loss only.

        Per the paper, WA negatives are the union of FA negatives in the batch.
        So neg_dataloader is still consumed — the miner needs both sub-batches to
        construct valid WA negative indices. FA patches are included in the forward
        pass so that WA negative embeddings are available. However, fa_indices are
        discarded after collation: loss_fa is never computed or backpropagated.

        Args:
            all_pos_dataloader: loader of anomaly-free patches (WA anchor/positive source).
                Iterated once per epoch — defines epoch length.
            neg_dataloader: loader of FA tuples, consumed round-robin to supply
                FA negatives for WA tuple construction. Not trained on directly.
        """
        self.model.train()
        assert self.single_batch, "train_epoch_wa requires single_batch=True (SimpleMiner + copy_negatives=False)"

        total_mined_loss = 0
        num_batches = 0

        pbar = tqdm(enumerate(all_pos_dataloader), total=len(all_pos_dataloader))
        neg_dataloader_iterator = patch3_dloaders.infinite_loader(neg_dataloader)

        for _, batch_mined in pbar:
            self.miner.ema_update(self.model, self.writer, self.global_step)

            batch_fa = next(neg_dataloader_iterator)

            # Mine: attaches FA negatives to WA anchor/positive index tuples.
            # mined_indices will have negative positions pre-set to Bm + fa_negative_idx.
            # fa_indices are produced but will be discarded below.
            (mined_patches, mined_indices, emb_mined_ema, _), \
            (fa_patches, fa_indices, _, _) = self.miner.mine(
                batch_mined['patches'], batch_mined['valid_indices'],
                batch_fa['patches'], batch_fa['valid_indices'])

            assert emb_mined_ema is None, "train_epoch_wa does not support EMA miners"

            # Collate WA + FA into one tensor. collate_ms_batches offsets fa_indices
            # by len(mined_patches); mined_indices are returned unchanged (negatives
            # already point into FA region). fa_indices_concat is intentionally discarded.
            patches, mined_indices_concat, _ = patch3_dloaders.collate_ms_batches(
                mined_patches, mined_indices, fa_patches, fa_indices)

            patches = patches.to(self.device)
            mined_indices_concat = tuple(t.to(self.device) for t in mined_indices_concat)

            # Single forward pass over combined batch (WA positives + FA negatives).
            # Third output (z_fa) is discarded — we only need z_mined.
            emb_all, z_mined, _ = self.model(patches)

            # WA loss only. FA patches are present as negatives for WA anchors,
            # but fa_indices are not used — no FA loss is computed.
            loss_mined = self.ms_loss_mined(z_mined, labels=None, indices_tuple=mined_indices_concat)

            self.optimizer.zero_grad()
            loss_mined.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_mined_loss += loss_mined.item()
            num_batches += 1
            self.global_step += batch_mined['patches'].shape[0]

        self.log_gradients(self.global_step)

        total_mined_loss /= num_batches
        self.writer.log_scalar('MinedLoss/Training', total_mined_loss, self.global_step)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.log_scalar('LearningRate/Current', current_lr, self.global_step)

        self.epoch += 1
        return {
            'train_sum_loss': total_mined_loss,
            'train_mined_loss': total_mined_loss,
            'train_forced_loss': 0,
        }

    def train_epoch(self, all_pos_dataloader, neg_dataloader):
        """Trains a single epoch of the contrastive encoder.
        Args:
            all_pos_dataloader: a loader that loads only positive (no anomalies) patches; a base for WA tuples
            neg_dataloader: a loader that loads only FA tuples
        """
        self.model.train()

        loss_mined = None
        loss_fa = None
        total_sum_loss = 0
        total_mined_loss = 0
        total_forced_loss = 0

        avg_patches = 0
        avg_mined_positive_pairs = 0
        avg_mined_negative_pairs = 0
        avg_fa_positive_pairs = 0
        avg_fa_negative_pairs = 0
        avg_forced_pairs = 0

        num_batches = 0

        pbar = tqdm(enumerate(all_pos_dataloader), total=len(all_pos_dataloader))
        neg_dataloader_iterator = patch3_dloaders.infinite_loader(neg_dataloader)
        for _, batch_mined in pbar:
            # For each epoch, we make a full loop over the WA tuples. FA tuples are loaded in a round-robin fashion

            # Update miner weights, if required
            self.miner.ema_update(self.model, self.writer, self.global_step)

            # Fetch a next batch of FA tuples
            batch_fa = next(neg_dataloader_iterator)

            _, _, _, loss_mined, loss_fa, mined_indices, fa_indices, Bm, Bfa = \
                self.batch_forward(batch_mined, batch_fa)

            if self.use_grad_norm:
                loss_weights = self.loss_weighter.compute_weights([loss_mined, loss_fa])
            else:
                loss_weights = {'WA': self.wa_loss_weight, 'FA': self.fa_loss_weight}
            loss = loss_mined*loss_weights['WA'] + loss_fa*loss_weights['FA']

            self.optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Do param optimization - a single step
            self.optimizer.step()

            # Log the number of positive / negative pairs
            avg_patches += Bm+Bfa
            avg_mined_positive_pairs += mined_indices[0].shape[0]
            avg_mined_negative_pairs += mined_indices[2].shape[0]
            avg_fa_positive_pairs += fa_indices[0].shape[0]
            avg_fa_negative_pairs += fa_indices[2].shape[0]

            total_sum_loss += loss.item()
            total_mined_loss += loss_mined.item()
            total_forced_loss += loss_fa.item()
            num_batches += 1
            # Add the number of fully aligned tuples we've seen so far as a global step
            self.global_step += Bm+Bfa

        self.log_gradients(self.global_step)

        # Get the mean of the loss, and update the loss weights
        total_sum_loss /= num_batches
        total_mined_loss /= num_batches
        total_forced_loss /= num_batches
        if self.use_grad_norm:
            self.loss_weighter.update_loss_history(torch.Tensor([total_mined_loss, total_forced_loss]).to(self.device))
            # Log gradient weights and similarities
            self.writer.log_scalars('LossGradSimilarities', self.loss_weighter.get_current_gradient_similarities(), self.global_step)
            self.writer.log_scalars( 'LossWeight', self.loss_weighter.get_current_weights(), self.global_step)

        avg_patches /= num_batches
        avg_mined_positive_pairs /= num_batches
        avg_mined_negative_pairs /= num_batches
        avg_fa_positive_pairs /= num_batches
        avg_fa_negative_pairs /= num_batches
        avg_forced_pairs /= num_batches

        self.writer.log_scalars(
            'InputData',
            {
                "patches": math.log10(avg_patches+1),
                "mined_positive_pairs": math.log10(avg_mined_positive_pairs+1),
                "mined_negative_pairs": math.log10(avg_mined_negative_pairs+1),
                "fa_positive_pairs": math.log10(avg_fa_positive_pairs+1),
                "fa_negative_pairs": math.log10(avg_fa_negative_pairs + 1)
            },
            self.global_step)


        self.writer.log_scalar('SumLoss/Training', total_sum_loss, self.global_step)
        self.writer.log_scalar('MinedLoss/Training', total_mined_loss, self.global_step)
        self.writer.log_scalar('ForcedLoss/Training', total_forced_loss, self.global_step)
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.log_scalar('LearningRate/Current', current_lr, self.global_step)

        self.epoch += 1
        return {
            'train_sum_loss': total_sum_loss,
            'train_mined_loss': total_mined_loss,
            'train_forced_loss': total_forced_loss
        }

    def validate(self, all_pos_dataloader, neg_dataloader, max_samples=100):
        self.model.eval()
        total_sum_loss = 0
        total_mined_loss = 0
        total_forced_loss = 0

        num_batches = 0
        positive_distances = []
        negative_distances = []

        embeddings = []
        labels = []
        metadata = []
        sample_count = 0

        batch_stats = BatchStats()

        neg_dataloader_iterator = patch3_dloaders.infinite_loader(neg_dataloader)
        with torch.no_grad():
            for batch_mined in all_pos_dataloader:
                batch_fa = next(neg_dataloader_iterator)

                emb_all, emb_mined, emb_fa, loss_mined, loss_fa, mined_indices_concat, fa_indices_concat, Bm, Bt = \
                    self.batch_forward(batch_mined, batch_fa)

                if self.use_grad_norm:
                    loss_weights = self.loss_weighter.get_current_weights()
                else:
                    loss_weights = {'WA': self.wa_loss_weight, 'FA': self.fa_loss_weight}
                loss = loss_mined*loss_weights['WA'] + loss_fa*loss_weights['FA']

                total_sum_loss += loss.item()
                total_mined_loss += loss_mined.item()
                total_forced_loss += loss_fa.item()

                num_batches += 1

                if self.single_batch:
                    anchor_emb = emb_all[fa_indices_concat[0]]
                    positive_emb = emb_all[fa_indices_concat[1]]
                    negative_emb = emb_all[fa_indices_concat[3]]
                    positive_distances, negative_distances = self.update_distance_stats(
                        anchor_emb, positive_emb, negative_emb, positive_distances, negative_distances)
                    batch_stats.update_1forward(emb_all, mined_indices_concat, fa_indices_concat)
                else:
                    anchor_emb = emb_fa[fa_indices_concat[0]]
                    positive_emb = emb_fa[fa_indices_concat[1]]
                    negative_emb = emb_fa[fa_indices_concat[3]]
                    positive_distances, negative_distances = self.update_distance_stats(
                        anchor_emb, positive_emb, negative_emb, positive_distances, negative_distances)
                    batch_stats.update_2forward(emb_mined, mined_indices_concat, emb_fa, fa_indices_concat)

            total_sum_loss /= num_batches
            total_mined_loss /= num_batches
            total_forced_loss /= num_batches
            # Update the scheduler. Use validation loss for that!
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_sum_loss)
            else:
                self.scheduler.step()

            mean_pos_distance = np.mean(positive_distances)
            mean_neg_distance = np.mean(negative_distances)
            distance_margin = mean_neg_distance - mean_pos_distance

            # Log validation metrics to TensorBoard
            self.writer.log_scalar('SumLoss/Validation', total_sum_loss, self.global_step)
            self.writer.log_scalar('MinedLoss/Validation', total_mined_loss, self.global_step)
            self.writer.log_scalar('ForcedLoss/Validation', total_forced_loss, self.global_step)
            self.writer.log_scalars('Distance/FADistance', {
                'Positive': mean_pos_distance,
                'Negative': mean_neg_distance,
                'Margin': distance_margin
            }, self.global_step, log_together=True)
            # Log validation histograms
            self.writer.log_histogram('Distance/FA_Positive_Distribution', np.array(positive_distances),
                                      self.global_step)
            self.writer.log_histogram('Distance/FA_Negative_Distribution', np.array(negative_distances),
                                      self.global_step)
            batch_stats_metrics = batch_stats.aggregate_and_report(self.writer, self.global_step)

            # Save the checkpoint
        val_metrics = {
            'val_sum_loss': total_sum_loss,
            'val_mined_loss': total_mined_loss,
            'val_forced_loss': total_forced_loss,
        } | batch_stats_metrics
        self.save_checkpoint(f'weights_epoch{self.epoch}', val_metrics)
        return val_metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    timestamp = UnifiedLogger.get_timestamp()

    effective_run_id = cfg.experiment.run_id if cfg.experiment.run_id else timestamp
    wandb_config = to_wandb_config(cfg)
    logger.warning("Input configuration:")
    logger.warning(OmegaConf.to_yaml(cfg))

    if cfg.experiment.use_wandb:
        wandb.init(
            project='cossad',
            name=f'{cfg.experiment.run_name}_{effective_run_id}_{cfg.data.split}',
            tags=cfg.experiment.tags,
            config=wandb_config
        )

    if cfg.data.split is None:
        logger.error('data split has not been specified. Can\'t train the model without data' )
        return

    train_files = []
    val_files = []
    test_files = []
    all_negative_patches_val = []
    all_negative_patches_train = []

    from omegaconf import ListConfig
    if isinstance(cfg.data.dataset, ListConfig):
        datasets = list(cfg.data.dataset)
    else:
        datasets = [cfg.data.dataset]  # wrap string into a list
    for dataset in datasets:
        allowed_classes_desc = f'{dataset}_{cfg.data.split}'
        patches_path = f'{cfg.data.patches_path}_{dataset}_{cfg.data.rotation}'
        # This allows to do proper K-fold validation based on the object type.
        allowed_objects = util_dloaders.get_object_classes_names(allowed_classes_desc)
        train_files_split, test_files_split, val_files_split = (
            dataloaders.util_dataloaders.prepare_splits(
                patches_path, 0.80, 0.00, 0.2, allowed_objects=allowed_objects))
        train_files.extend(train_files_split)
        test_files.extend(test_files_split)
        val_files.extend(val_files_split)

        # Load negative patches for a dataset. NOTE: these are not used with a simple miner.
        neg_train, neg_val = patch3_dloaders.load_all_negatives(patches_path, cfg.data.points_per_patch, 0.80)
        all_negative_patches_train.append(neg_train)
        all_negative_patches_val.append(neg_val)

        logger.info(f'Data loaded from {dataset} dataset: {len(train_files)} training samples, {len(val_files)} validation samples')
    assert len(train_files) != 0, 'No training data could beloaded'
    logger.info(f'Data loaded froom all datasets: {len(train_files)} training samples, {len(val_files)} validation samples')
    negatives_val = torch.cat(all_negative_patches_val, dim=0)
    negatives_train = torch.cat(all_negative_patches_train, dim=0)
    logger.info(
        f'Loaded negative patches from all datasets: {negatives_train.shape[0]} training samples, '
        f'{negatives_val.shape[0]} validation samples')

    transformsPrePP = v2.Compose([
        RandomCutTransform(min_cuts=1, max_cuts=4, min_cut_r=0.05, max_cut_r=0.15, max_percent_pts_cut=0.25)
    ])
    transformsPostPP = v2.Compose([
        RandomRotationTransform(angle_range_deg=(-90, 90), same_rotation=True),
        RandomRotationTransform(angle_range_deg=(-3, 3), same_rotation=False),
        RandomTranslationTransform(translation_range=0.02, same_translation=True),
        GaussianNoiseTransform(sigma=0.003, clip=0.01),
    ])

    # train_dataset = patch3_dloaders.MultiSimilarityDataset(
    #     train_files, centering=constants.TripletCentereing(cfg.data.centering),
    #     num_points=cfg.data.points_per_patch,
    #     max_good_patches_per_class=cfg.data.max_good_patches_per_class,
    #     num_positive_files_for_single_negative=cfg.data.positive_per_negative,
    #     transformsPostPP=transformsPostPP, transformsPrePP=transformsPrePP)
    # val_dataset = patch3_dloaders.MultiSimilarityDataset(
    #     val_files,  centering=constants.TripletCentereing(cfg.data.centering),
    #     max_good_patches_per_class=cfg.data.max_good_patches_per_class,
    #     num_positive_files_for_single_negative=cfg.data.positive_per_negative,
    #     num_points=cfg.data.points_per_patch)

    train_all_pos_dataset = patch3_dloaders.MultiSimilarityAllPositiveDataset(
        train_files, centering=constants.TripletCentereing(cfg.data.centering), num_points=cfg.data.points_per_patch,
        transformsPostPP=transformsPostPP, transformsPrePP=transformsPrePP,
        max_good_patches_per_class=cfg.data.max_good_per_positive_anchor)
    val_all_pos_dataset = patch3_dloaders.MultiSimilarityAllPositiveDataset(
        val_files, centering=constants.TripletCentereing(cfg.data.centering), num_points=cfg.data.points_per_patch,
        transformsPostPP=transformsPostPP, transformsPrePP=transformsPrePP,
        max_good_patches_per_class=cfg.data.max_good_per_positive_anchor)
    train_neg_dataset = patch3_dloaders.MultiSimilarityNegTupleDataset(
        train_files, centering=constants.TripletCentereing(cfg.data.centering),
        transformsPostPP=transformsPostPP, transformsPrePP=transformsPrePP, num_points=cfg.data.points_per_patch,
        max_good_patches_per_class=cfg.data.max_good_per_negative_anchor)
    val_neg_dataset = patch3_dloaders.MultiSimilarityNegTupleDataset(
        val_files, centering=constants.TripletCentereing(cfg.data.centering),
        transformsPostPP=transformsPostPP, transformsPrePP=transformsPrePP, num_points=cfg.data.points_per_patch,
        max_good_patches_per_class=cfg.data.max_good_per_negative_anchor)

    #
    # train_dataset = patch3_dloaders.MultiSimilarityDataset(
    #     train_files, centering=constants.TripletCentereing(cfg.data.centering),
    #     num_points=cfg.data.points_per_patch, transformsPostPP=transformsPostPP, transformsPrePP=transformsPrePP)
    # val_dataset = patch3_dloaders.DatasetPatchTriplet(
    #     val_files,  centering=constants.TripletCentereing(cfg.data.centering), num_points=cfg.data.points_per_patch)

    # Create dataloaders
    num_workers = 0 if in_debugger() else 6

    # Create model
    main_device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = instantiate_model(cfg.model.name, cfg.model.embedding_dim, cfg.model.use_dropout)
    model.to(main_device)

    # Init the patch miner
    if cfg.triplet_mining.miner == 'Simple':
        # This is the working functionality as of 01/2026 (ICPR)
        miner = SimpleMiner(copy_negatives=cfg.experiment.copy_negatives)
    else:
        miner_network = deepcopy(model)  # identical architecture + weights + buffers
        miner_network.eval()  # keep it always in eval
        for p in miner_network.parameters():
            p.requires_grad = False  # make sure no grads leak into miner
        # miner_device = torch.device(f"cuda:{1 if torch.cuda.device_count() > 1 else 0}")
        miner_device = torch.device(f"cuda:0")
        miner = None
        if cfg.triplet_mining.miner == 'EMA':
            miner = FullEMAMiner(miner_network.to(miner_device), negatives_train,
                                 num_to_mine=cfg.triplet_mining.num_to_mine,
                                 momentum=cfg.triplet_mining.mining_momentum)
        elif cfg.triplet_mining.miner == 'SimpleEMA':
            miner = SimpleEMAMiner(
                miner_network.to(miner_device), negatives_train,
                num_to_mine=cfg.triplet_mining.num_to_mine, momentum=cfg.triplet_mining.mining_momentum)
        elif cfg.triplet_mining.miner == 'SimpleRandom':
            miner = SimpleRandomMiner(
                miner_network.to(miner_device), negatives_train,
                num_to_mine=cfg.triplet_mining.num_to_mine, momentum=cfg.triplet_mining.mining_momentum)
        assert miner is not None, f'Could not initialize miner with name {cfg.triplet_mining.miner}'
        miner.negatives_update()

    train_all_pos_loader = DataLoader(
        train_all_pos_dataset, batch_size=cfg.training.all_pos_batch_size, shuffle=True, num_workers=num_workers,
        # pin_memory=True, persistent_workers=True,
        drop_last=True, collate_fn=patch3_dloaders.AllPositiveTupleCollator())
    val_all_pos_loader = DataLoader(
        val_all_pos_dataset, batch_size=cfg.training.all_pos_batch_size, shuffle=True, num_workers=num_workers,
        # pin_memory=True, persistent_workers=True,
        drop_last=True, collate_fn=patch3_dloaders.AllPositiveTupleCollator())
    train_neg_loader = DataLoader(
        train_neg_dataset, batch_size=cfg.training.neg_batch_size, shuffle=True, num_workers=num_workers,
        # pin_memory=True, persistent_workers=True,
        drop_last=True, collate_fn=patch3_dloaders.SingleNegativeTupleCollator())
    val_neg_loader = DataLoader(
        val_neg_dataset, batch_size=cfg.training.neg_batch_size, shuffle=True, num_workers=num_workers,
        # pin_memory=True, persistent_workers=True,
        drop_last=True, collate_fn=patch3_dloaders.SingleNegativeTupleCollator())

    # Create trainer
    trainer = MSTrainer(main_device, model, cfg, timestamp=timestamp)
    # Set the miner
    trainer.set_miner(miner)

    # Initiate the "early stopper" that will monitor accuracy and stop training if it is not improving.
    stopper = EarlyStopping(cfg.training.early_stop_tolerance)
    best_fa_margin = -float('inf')
    best_mined_margin = -float('inf')

    training_succeeded = False
    try:
        for epoch in range(cfg.training.epochs):
            # Train
            # train_metrics = trainer.train_epoch(train_all_pos_loader, train_neg_loader)
            # train_metrics = trainer.train_epoch_fa(train_neg_loader)
            train_metrics = trainer.train_epoch_wa(train_all_pos_loader, train_neg_loader)
            # Validate
            val_metrics = trainer.validate(val_all_pos_loader, val_neg_loader)

            trainer.writer.log_scalars(
                'SumLoss/Combined',
                {"val": val_metrics["val_sum_loss"], "train": train_metrics["train_sum_loss"]},
                trainer.global_step)
            trainer.writer.log_scalars(
                'MinedLoss/Combined',
                {"val": val_metrics["val_mined_loss"], "train": train_metrics["train_mined_loss"]},
                trainer.global_step)
            trainer.writer.log_scalars(
                'ForcedLoss/Combined',
                {"val": val_metrics["val_forced_loss"], "train": train_metrics["train_forced_loss"]},
                trainer.global_step)

            print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
            print(f"  Train Loss: {train_metrics['train_sum_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_sum_loss']:.4f}")
            print(f"  FA distance Margin: {val_metrics['mean_margin_fa']:.4f}")
            print(f"  Mined distance Margin: {val_metrics['mean_margin_mined']:.4f}")

            # Save best model based on distance margin
            if val_metrics['mean_margin_fa'] > best_fa_margin:
                best_fa_margin = val_metrics['mean_margin_fa']
                trainer.save_checkpoint(f'best_{cfg.model.name}.tmp')
                print(f"  Saved new best model with margin: {best_fa_margin:.4f}")
            if val_metrics['mean_margin_mined'] > best_mined_margin:
                best_mined_margin = val_metrics['mean_margin_mined']
            trainer.writer.log_scalar('Similarity/MaxFAMargin', best_fa_margin, trainer.global_step)
            trainer.writer.log_scalar('Similarity/MaxMinedMargin', best_mined_margin, trainer.global_step)

            if stopper(val_metrics['val_sum_loss'], val_metrics['mean_margin_fa']):
                logger.warning('Network not improving - stopping training early')
                break

            print("-" * 50)
        training_succeeded = True
    except BaseException:
        logger.exception("Training failed with an unhandled exception")
    finally:
        if training_succeeded:
            tmp_path = os.path.join(trainer.weights_dir, f'best_{cfg.model.name}.tmp')
            pth_path = os.path.join(trainer.weights_dir, f'best_{cfg.model.name}.pth')
            if os.path.exists(tmp_path):
                os.rename(tmp_path, pth_path)
    if cfg.experiment.use_wandb:
        wandb.finish()

    return 0 if training_succeeded else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
