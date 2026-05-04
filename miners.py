"""Miners accept sub-batches of FA and WA tuples, combine them into a shape ingestible by a contrastive learner.
 Optionally do tuple mining"""

import logging
from copy import deepcopy
from typing import Tuple, NamedTuple
import numpy as np

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

#  Indices within the contrastive tuple, in the Python Metric Learning format
IndexTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class Mined(NamedTuple):
    """Represents the result of tuple mining."""
    # A batch of mined patches. This may include both patches from the original batch that were sent to a miner, as well
    # as patches that were mined.
    # This set should be send to a network as is
    patches: torch.Tensor
    # Final set of contrastive indices for contrastive learning. This should be the final set of indices that must be
    # sent to the network as is - except if we adjust indices when we merge FA and WA batches. Even if we return
    # embeddings, the corresponding indices should be already in this tuple.
    indices_tuple: IndexTuple
    # These are raw patch embeddings that we potentially mined using MoCo-trained encoder.
    # These should be concatenated with the batch  _after_ we received the embeddings from the network,
    # right before loss computation.
    embeddings : torch.Tensor | None = None
    # These are the embeddings for a specific task (i.e. if we are using MTL head).
    z_mined:  torch.Tensor | None = None


class MinerBase:
    def mine(self, mined_patches, mined_indices_tuple, fa_patches, fa_indices_tuple) -> Tuple[Mined, Mined]:
        """Mines patches or embeddings for contrastive learning.

        Args:
            mined_patches: a tensor of WA tuples.
            mined_indices_tuple: a tuple of index tensors, in a PythonMetricLearning format, that denotes correct
                "anchor -> positive" relations for the mined_patches tensor. NOTE: we expect that "anchor-negative"
                relations will be empty here.
            fa_patches: a tensor of FA tuples
            fa_indices_tuple: a tuple of index tensors, in a PythonMetricLearning format, for FA tuples. We expect
                that both ""anchor-positive" and "anchor-negative" relations are defined.


        Returns:
            a pair of Mined tuples. The first one refers to the mined patches, the second - to FA patches

        """
        raise NotImplementedError

    @staticmethod
    def get_positive_indices_in_mined_microbatch_(mined_patches, mined_indices_tuple) -> torch.Tensor:
        """Returns a tensor of indices of positive items in the FA sub-batch. Validates sub-batch integrity."""

        # Indices of all positive patches in the "all positive" sub-batch
        batch_positive_indices = torch.unique(mined_indices_tuple[0])
        batch_positive_indices, _ = batch_positive_indices.sort(descending=False)
        # Indices of the positive patches must be a continuous sequence in (0, num. positive patches)
        # Code above is defensive. we may remove this in prod, and use only arange-generated indices
        expected_pos_indices = torch.arange(batch_positive_indices.shape[0], device=batch_positive_indices.device)
        assert torch.equal(expected_pos_indices, batch_positive_indices), \
            "Positive anchors must be a contiguous range [0..P-1]"
        assert mined_patches.shape[0] == batch_positive_indices.shape[0], \
            'The number of mined patches  should be equal to the number of anchor indices'
        assert batch_positive_indices.device == mined_patches.device
        return batch_positive_indices

    def ema_update(self, trainer_model_in, writer, global_step):
        """If the miner is based on Exponential Moving Average network (MoCo style), update EMA weights here.

        Args:
            trainer_model_in: a trainer model, i.e. one that is updadted through gradient descent
            writer: a WB / tensorboard writer.
            global_step: the global training step (i.e. num. seen tuples)
        """
        pass

    def negatives_update(self):
        """If we maintain EMA-mined embeddings, update the mined embeddings here."""
        pass


class SimpleMiner(MinerBase):
    """ As of 01/2026, this is the miner used. Does not mine tuples per SE, rather creates a valid contrastive patch. """
    def __init__(self, copy_negatives=False):
        """
        Args:
            copy_negatives: copies the set of negatives from FA tuples into a tensor that holds WA tuples.
            Mostly for experimentation
        """
        super(SimpleMiner, self).__init__()
        # Copy a set of negative patches from "fully-aligned" tuples. Append these to mined_patches, i.e. "wa" tuples
        # TODO: check thooroughly
        self.copy_negatives = copy_negatives
        logging.warning('SimpleMiner is initialized')

    def mine(self, mined_patches, mined_indices_tuple, fa_patches, fa_indices_tuple) -> Tuple[Mined, Mined]:
        """Augments each patch in the "mined" microbatch - which are all positive patches - with a set of negattive
         patches extracted from the fully-aligned microbatch (i.e. some patches are negative).
         """
        assert fa_patches.device == mined_patches.device
        # Get indices of all the positives from "mined" tuples and verify they are correct.
        mined_positive_indices = MinerBase.get_positive_indices_in_mined_microbatch_(mined_patches, mined_indices_tuple)

        # Indices of the negative patches in the "fully aligned" sub-batch
        fa_negative_indices = torch.unique(fa_indices_tuple[3])
        # A start position of these patches. Right end the end of a tensor of "WA" positives
        # Should be the same regardless if we will append them, or simply re-use whatever is in the negative batch
        start_neg_idx = mined_patches.shape[0]

        # Get a copy of negative patches from the FA sub-batch, if you want to add these to a batch
        if self.copy_negatives:
            negative_patches = fa_patches[fa_negative_indices]
            # Append negatives to the mined patches
            mined_patches = torch.cat((mined_patches, negative_patches))
            mined_negative_indices = torch.arange(
                start_neg_idx, start_neg_idx + negative_patches.shape[0], device=mined_indices_tuple[0].device)
        else:
            # Re-use whatever negative indices are in the batch without copying corresponding patches
            start_neg_idx = mined_patches.shape[0]
            mined_negative_indices = start_neg_idx + fa_negative_indices

        # Indices of all valid pairs for the current batches.
        positive_negative_pairs_idx = torch.cartesian_prod(mined_positive_indices, mined_negative_indices)
        return (
            # mined sub-batch
            Mined(patches=mined_patches,
                  indices_tuple=(mined_indices_tuple[0], mined_indices_tuple[1], positive_negative_pairs_idx[:,0], positive_negative_pairs_idx[:,1]),
                  embeddings=None, z_mined=None),
            # FA sub-batch
            Mined(patches= fa_patches, indices_tuple=fa_indices_tuple, embeddings=None, z_mined=None)
        )


class EMAMinerBase(MinerBase):
    """Base class for miners that are doing EMA-based mining for contrastive learner.

    Runs the inference on a batch, and zeroes the loss on the items that are considered.
    Can either mine the most challenging patches using EMA network and append these to the batch,
    or even can append mined embeddings.
    """

    def __init__(self, miner_model, negatives, num_to_mine,
                 update_embeddings_every=10, momentum=0.99):# , margin, crit_yield, momentum=0.99):
        """
        Args:
            miner_model: a deep learning model that will be used for mining.
                Note: It's a MOCO-style miner, not the actual model being trained!
            negatives: a tensor [N, n_points, 3] of all negative patches
            margin: a margin used to compute if an example if semihard.
                If the d_ap - d_an < margin, this is a "semi-hard" example.
            crit_yield: a minimnum yield of a miner. A percentage of selected items vs. size of a minibatch
                If the percent of semi-hard is > crit_yield, only semi-hards are mined
                If the percent of semi-hard is < crit_yield, hards are mined additionally
                If the percent of (hard + semihard) < crit_yield, "easy" triplets are added till the desired yield
                    is achieved
        """
        super(EMAMinerBase, self).__init__()
        # self.main_device = main_device
        self.miner_model = miner_model
        self.num_to_mine = num_to_mine
        # self.margin = margin
        # self.num_hard = 0
        # self.num_semihard = 0
        # self.num_easy = 0
        # self.num_all = 0
        # self.crit_yield = float(crit_yield)
        # assert 0.0 <= self.crit_yield <= 1.0, "crit_yield must be in [0,1]"
        self.momentum = momentum
        # Get the device for a miner network
        self.miner_device = next(self.miner_model.parameters()).device

        # Set of negative patches
        assert negatives is not None and negatives.shape[0] != 0
        self.negatives = negatives.to(self.miner_device)
        # embeddings for the negative patches
        self.negatives_emb = None

        self.batch_size = 512
        if self.negatives.shape[0] <= self.batch_size:
            self.batch_size = self.negatives.shape[0]

        self.update_embeddings_every_batches = update_embeddings_every
        self.batch_counter = 0

    # def get_margin(self):
    #     return self.margin
    #
    # def set_margin(self, margin):
    #     self.margin = margin

    def _unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    @torch.no_grad()
    def ema_update(self, trainer_model_in, writer, global_step):
        """Update the parameters of the miner network in an EMA fashion."""

        # This is for debug purposes, when you don't add EMA-mined embeddings to the batch, but track their
        # similarity to the embeddings produced by the main model.
        self.log_ema_consistency(trainer_model_in, writer, global_step)

        # This is for the distributed training case
        trainer_model = self._unwrap(trainer_model_in)
        miner_named = dict(self.miner_model.named_parameters())
        trainer_named = dict(trainer_model.named_parameters())
        assert miner_named.keys() == trainer_named.keys(), "Miner and trainer param names differ"

        for name, p_trainer in trainer_named.items():
            p_mine = miner_named[name]
            p_mine.mul_(self.momentum).add_(p_trainer.to(self.miner_device), alpha=1 - self.momentum)

        mb_named = dict(self.miner_model.named_buffers())
        tb_named = dict(trainer_model.named_buffers())
        for name, b_main in tb_named.items():
            if name in mb_named:
                mb_named[name].copy_(b_main.to(self.miner_device))

    @torch.no_grad()
    def negatives_update(self):
        """Update embeddings for all the negatives present"""
        new_negative_batches = []
        # local_negatives = self.negatives.to(self.miner_device)
        N = self.negatives.shape[0]
        n_splits = N // self.batch_size
        neg_batches = torch.tensor_split(self.negatives, n_splits)
        self.miner_model.eval()
        with torch.no_grad():
            for batch in neg_batches:
                out_features, _, _ = self.miner_model(batch)
                new_negative_batches.append(out_features)
        negatives_emb = torch.cat(new_negative_batches, dim=0)
        assert negatives_emb.shape[0] == self.negatives.shape[0], \
            "negatives_emb and negatives must have the same length"
        self.negatives_emb = F.normalize(negatives_emb, dim=1)#.to(self.miner_device)
        # May help to empty GPU cache and save some memory

    @torch.no_grad()
    def log_ema_consistency(self, trainer_model_in, log_writer, global_step):
        """ Debugging code: check how much EMA encoder state lags behind main encoder by jointly logging their weights.
        """
        trainer_model = self._unwrap(trainer_model_in)

        # Parameter lag (expected to be non-zero due to EMA)
        param_diffs = []
        for (n1, p1), (n2, p2) in zip(
                self.miner_model.named_parameters(),
                trainer_model.named_parameters()
        ):
            diff = (p1 - p2.to(self.miner_device)).abs().mean()
            param_diffs.append(diff.item())

        # Buffer lag for BatchNorm (should match parameter lag pattern after fix)
        bn_mean_diffs = []
        bn_var_diffs = []
        for (n1, m1), (n2, m2) in zip(
                self.miner_model.named_modules(),
                trainer_model.named_modules()
        ):
            if isinstance(m1, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                mean_diff = (m1.running_mean - m2.running_mean.to(self.miner_device)).abs().mean()
                var_diff = (m1.running_var - m2.running_var.to(self.miner_device)).abs().mean()
                bn_mean_diffs.append(mean_diff.item())
                bn_var_diffs.append(var_diff.item())

        log_writer.log_scalars(
            'EMADebug',
            {
                'mean_param_diff': np.mean(param_diffs),
                'bn_mean_diffs': np.mean(bn_mean_diffs),
                'bn_var_diffs': np.mean(bn_var_diffs),
            },
            global_step
        )


class SimpleEMAMiner(EMAMinerBase):
    """This is an experimental code that runs EMA encoder on a FA sub-batch, get embeddings of negatives
     and appends these (not raw patches) to a WA sub-batch. Training loss will be computed on mined embeddings.
    """
    def __init__(self, miner_model, negatives, num_to_mine, update_embeddings_every=10, momentum=0.99):# , margin, crit_yield, momentum=0.99):
        super(SimpleEMAMiner, self).__init__(miner_model, negatives, num_to_mine, update_embeddings_every, momentum)
        logging.warning('SimpleEMAMiner is initialized')

    @torch.no_grad()
    def mine(self, mined_patches, mined_indices_tuple, fa_patches, fa_indices_tuple) -> Tuple[Mined, Mined]:
        assert fa_patches.device == mined_patches.device
        # Get indices of all the positives from "mined" tuples and verify they are correct.
        mined_positive_indices = MinerBase.get_positive_indices_in_mined_microbatch_(mined_patches, mined_indices_tuple)
        # Indices of the negative patches in the "triplet" sub-batch
        triplet_negative_indices = torch.unique(fa_indices_tuple[3])
        # Index where we will add mined positive embeddings
        start_neg_idx = mined_patches.shape[0]

        # Here we get the embeddings of the negative patches in FA sub-batch.
        negative_patches = fa_patches[triplet_negative_indices]
        self.miner_model.eval()
        emb_mined_ema, z_mined_ema, _ = self.miner_model(negative_patches.to(self.miner_device))
        emb_mined_ema = emb_mined_ema.detach().to(mined_patches.device)
        assert emb_mined_ema.shape[0] == negative_patches.shape[0]
        assert z_mined_ema.shape[0] == negative_patches.shape[0]

        # # TODO(alexta): investigation / debug only. Here we add back patches from FA tuples, but keep mined embeddings.
        # # Later we will check - how mined embeddings are similar to the embeddings by the trainer network.
        # negative_patches = fa_patches[triplet_negative_indices]
        # mined_patches = torch.cat((mined_patches, negative_patches))

        mined_negative_indices = torch.arange(
            start_neg_idx, start_neg_idx + z_mined_ema.shape[0], device=mined_indices_tuple[0].device)
        positive_negative_pairs_idx = torch.cartesian_prod(mined_positive_indices, mined_negative_indices)

        return (
            Mined(patches=mined_patches,
                  indices_tuple=(mined_indices_tuple[0], mined_indices_tuple[1], positive_negative_pairs_idx[:,0], positive_negative_pairs_idx[:,1]),
                  embeddings=emb_mined_ema, z_mined=z_mined_ema),
            Mined(patches= fa_patches, indices_tuple=fa_indices_tuple, embeddings=None, z_mined=None)
        )


class SimpleRandomMiner(EMAMinerBase):
    """This is an experimental code that adds the random negative patches to the minibatch of mined patches (no EMA).

    Expected to fail, used to experimentally validate our batch structure, based on gradients of MS loss.
    """
    def __init__(self, miner_model, negatives, num_to_mine, update_embeddings_every=10, momentum=0.99):# , margin, crit_yield, momentum=0.99):
        super(SimpleRandomMiner, self).__init__(miner_model, negatives, num_to_mine, update_embeddings_every, momentum)
        logging.warning('SimpleRandomMiner is initialized')

    @torch.no_grad()
    def mine(self, mined_patches, mined_indices_tuple, fa_patches, fa_indices_tuple) -> Tuple[Mined, Mined]:
        assert fa_patches.device == mined_patches.device
        # Get indices of all the positives from "mined" tuples and verify they are correct.
        mined_positive_indices = MinerBase.get_positive_indices_in_mined_microbatch_(mined_patches, mined_indices_tuple)
        # Index where we will add mined positive embeddings
        start_neg_idx = mined_patches.shape[0]

        # Get the number of negatives we need to add; it should be the same as the number of FA negatives.
        fa_negative_indices = torch.unique(fa_indices_tuple[3])
        M = fa_negative_indices.shape[0]

        # Randomly get a number of negative patches we need
        N = self.negatives.shape[0]
        idx_negatives = torch.randperm(N, device=self.negatives.device)[:M]
        negative_patches = self.negatives[idx_negatives].to(mined_patches.device)

        # Append negatives to the mined patches
        mined_patches = torch.cat((mined_patches, negative_patches))
        mined_negative_indices = torch.arange(
            start_neg_idx, start_neg_idx + negative_patches.shape[0], device=mined_indices_tuple[0].device)
        positive_negative_pairs_idx = torch.cartesian_prod(mined_positive_indices, mined_negative_indices)

        return (
            # mined sub-batch
            Mined(patches=mined_patches,
                  indices_tuple=(mined_indices_tuple[0], mined_indices_tuple[1], positive_negative_pairs_idx[:,0], positive_negative_pairs_idx[:,1]),
                  embeddings=None, z_mined=None),
            # FA sub-batch
            Mined(patches= fa_patches, indices_tuple=fa_indices_tuple, embeddings=None, z_mined=None)
        )


class FullEMAMiner(MinerBase):
    """ Mines the hard negatives from the existing tensor of negatives.

    TODO(alexta): as of 01/26, it is not used. Proper integration of mined patches into a batch is a future work.
    """
    def __init__(self, miner_model, negatives, num_to_mine, update_embeddings_every=10, momentum=0.99):# , margin, crit_yield, momentum=0.99):
        super(FullEMAMiner, self).__init__(miner_model, negatives, num_to_mine, update_embeddings_every, momentum)

    @torch.no_grad()
    def mine(self, mined_patches, mined_indices_tuple, fa_patches, fa_indices_tuple) -> Tuple[torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Mines the hard negatives from the existing tensor of negatives.

        Args:
            mined_patches: a tensor [P, num_points, 3] of all positive patches.
                These patches are only "positive" patches, there are no negative patches in this microbatch.
            mined_indices_tuple: an (unfinished) tuple of positive indexes, "PyTorch Metric Learning" style.
                contains only indices of a "template to positive pairs".
            fa_patches: a microbatch of fully-algined tuples. Not used.
            fa_indices_tuple: indices of fully aligned "positive vs negative" tuples. Not used.
        Returns:
            patches: an updated patch tensor, where we will append mined negatives
            positive_indices_tuple: an updated tuple, where for each positive index we will have "PML"-style
                index of negative
            triplet_indices_tuple: indices of a "positive cs negative" tuples.
        """
        self.batch_counter += 1
        # Actual code
        # if self.batch_counter % self.update_embeddings_every_batches == 0:
        #     self.negatives_update()

        assert self.negatives_emb is not None
        # assert mined_patches.device == self.main_device
        mined_positive_indices = MinerBase.get_positive_indices_in_mined_microbatch_(mined_patches, mined_indices_tuple)
        assert mined_positive_indices.device == mined_patches.device

        N, D = self.negatives_emb.shape
        P = mined_patches.shape[0]
        assert torch.max(mined_positive_indices)+1 == P
        self.miner_model.eval()

        with torch.no_grad():  # <— put this back
            # Get a random batch of negatives with a shape [N, D]
            # idx = torch.randperm(N, device=patches.device)[:self.batch_size]
            # batch = self.negatives[idx]
            # neg_emb = self.miner_model(batch)

            # positives_emb, _, _ = self.miner_model(mined_patches.to(self.miner_device))
            # # Normalize embeddings to make sure cosine similarity makes sense
            # positives_emb = F.normalize(positives_emb, dim=1)   #.to(mined_patches.device)
            # assert positives_emb.shape[1] == D
            # assert positives_emb.shape[0] == P
            #
            # # Compute the similarity between the positive and negative embeddings. The result should be of shape [P, N]
            # sim_pos_neg = positives_emb @ self.negatives_emb.T
            #
            # # Get the top K negative patches that are most similar (i.e. hardest negative pairs) to each positive patch.
            # assert self.num_to_mine <= N

            # _, idx = torch.topk(sim_pos_neg, k=self.num_to_mine, dim=1) # idx shoulld be of shape [P, K]
            # # TODO(alexta): check this works as expected. Negatives should be a tensor of [N, n_points, 3]
            # hardest_negatives = self.negatives[idx]  # shape [P, K, n_points, 3]
            # # Align negative patches sequentially, in the order of the corresponding positive patches.
            # # I.e. first there are K negative patches for the 1st positive patch, then K negative patches for the 2nd
            # # positive patch and so on. The final shape should be shape [P*K, n_points, 3]
            # hardest_negatives = torch.reshape(
            #     hardest_negatives, (-1, self.negatives.shape[-2], self.negatives.shape[-1]))


            # Debugging: select negatives randomly
            num_negatives = P*self.num_to_mine
            # num_negatives = triplet_patches.shape[0] // 4
            idx = torch.randperm(N, device=self.negatives.device)[:num_negatives]
            hardest_negatives = self.negatives[idx]

            #  Debug: the same code as the dataloader
            # # Add negative view to the positive microbatch and generate indices for these
            # mined_patches = torch.cat((mined_patches, hardest_negatives.to(mined_patches.device)), dim=0)
            # mined_negative_indices = torch.arange(P, P + hardest_negatives.shape[0], device=mined_indices_tuple[0].device)
            # positive_negative_pairs_idx = torch.cartesian_prod(mined_positive_indices, mined_negative_indices)
            # positives_idx = positive_negative_pairs_idx[:,0]
            # negatives_idx = positive_negative_pairs_idx[:,1]

            # Actual code
            # Add the mined negatives to the vector of positives.
            mined_patches = torch.cat((mined_patches, hardest_negatives.to(mined_patches.device)))
            # Create an index of positive-to-negative pairs.
            # Positive "anchors" create an interleaved index. E.g. for K=3: 0, 0, 0, 1, 1, 1, ..., P-1, P-1, P-1
            positives_idx = torch.arange(P, device=mined_patches.device).repeat_interleave(self.num_to_mine)
            # Corresponding negatives: just a sequence starting from B (i.e. starting index of a 1st hard negative)
            negatives_idx = torch.arange(P, P+P*self.num_to_mine, device=mined_patches.device)
        return (mined_patches,
                (mined_indices_tuple[0], mined_indices_tuple[1], positives_idx, negatives_idx),
                fa_patches,
                fa_indices_tuple)
