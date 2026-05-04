# import external.risurconv.models.risurconv_cls as riconvmodel
import external.riconv2.models.riconv2_cls as riconvmodel

import torch
import torch.nn as nn
import torch.nn.functional as F
from external.riconv2.models.riconv2_utils import compute_LRA

class RISurConvEncoder(nn.Module):
    # def __init__(self, embedding_dim=64, normal_channel=False, normalize_embeddings=True):
    def __init__(self, rep_head, mtl_head=None, normal_channel=True, normalize_embeddings=True):
        super(RISurConvEncoder, self).__init__()
        self.normal_channel = normal_channel

        # self.backbone = riconvmodel.get_model(num_class=embedding_dim, n=1, normal_channel=normal_channel)
        self.backbone = riconvmodel.get_model(n=1, normal_channel=normal_channel)
        self.rep_head = rep_head
        self.mtl_head = mtl_head
        # self.head = head

        # # alexta: this is a copycat from encoders_pointnet
        # self.head1 = nn.Linear(512, 256)
        # self.head2 = nn.Linear(256, embedding_dim)
        #
        # self.dropout = nn.Dropout(p=0.4)
        # # self.bn1 = nn.BatchNorm1d(512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # # Optional: Add L2 normalization for better contrastive learning
        self.normalize_embeddings = normalize_embeddings

    def forward(self, xyz):
        # seems that for a RIConvNet we should have [B, N, 3]
        # if xyz.dim() == 3:
        #     xyz = xyz.transpose(2, 1)  # [B, N, 3] -> [B, 3, N]
        device = next(self.parameters()).device
        norm = torch.zeros_like(xyz).to(device)

        # Alexta: this is a copy from the ModelNetDataLoader for RIConv++ code
        B = 16
        L = int(len(xyz) / B)
        for index in range(L):
            norm[index * B:(index + 1) * B, :, :] = compute_LRA(xyz[index * B:(index + 1) * B, :, :],
                                                                True, nsample=32)
        norm[(index + 1) * B:, :, :] = compute_LRA(xyz[(index + 1) * B:, :, :], True, nsample=32)
        xyz_pseudonormals = torch.cat([xyz, norm], dim=-1)

        # zeros = torch.zeros_like(xyz)
        # xyz_with_fake_normals = torch.cat([xyz, zeros], dim=2)

        logits = self.backbone(xyz_pseudonormals)
        rep_embeddings = self.rep_head(logits)
        if self.mtl_head is not None:
            z_mined, z_forced = self.mtl_head(rep_embeddings)
        else:
            z_mined, z_forced = rep_embeddings, rep_embeddings

        if self.normalize_embeddings:
            embeddings = F.normalize(rep_embeddings, p=2, dim=1)
        else:
            embeddings = rep_embeddings
        return embeddings, z_mined, z_forced
