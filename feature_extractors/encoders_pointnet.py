import external.pointnet2.models.pointnet_cls as pmodel
import external.pointnet2.models.pointnet2_cls_msg as p2_msg_model
import external.pointnet2.models.pointnet2_cls_ssg as p2_ssg_model

import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_dim=64, normal_channel=False):
        super(PointNetEncoder, self).__init__()
        # This doesn't matter. We'll cut the head of the model backbone and use only features
        self.num_class = 10
        self.normal_channel = normal_channel

        # Use yanx27's PointNet++ model as backbone
        # We'll modify their classification model to output embeddings
        self.backbone = pmodel.get_model(k=embedding_dim, normal_channel=normal_channel)

        # Optional: Add L2 normalization for better contrastive learning
        self.normalize = True

    def forward(self, xyz):
        """
        Forward pass
        Args:
            xyz: Point cloud tensor [B, N, 3] or [B, N, 6] (with normals)
        Returns:
            embeddings: [B, embedding_dim]
        """
        # PointNet++ expects input in format [B, C, N] where C is channels (3 for xyz)
        if xyz.dim() == 3:
            xyz = xyz.transpose(2, 1)  # [B, N, 3] -> [B, 3, N]

        # Get embeddings from backbone. Sould be [B, C, 1]
        pnet2_embeddings, _ = self.backbone(xyz)
        embeddings = pnet2_embeddings

        # L2 normalize embeddings for better contrastive learning
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class PointNet2Encoder(nn.Module):
    """
    PointNet++ encoder for triplet learning
    Extracts embeddings from point clouds
    """

    def __init__(self, embedding_dim=256, normal_channel=False, pnet2_version='pointnet2_ssg',
                 normalize_embeddings=True):
        super(PointNet2Encoder, self).__init__()
        # This doesn't matter. We'll cut the head of the model backbone and use only features
        self.num_class = 10
        self.normal_channel = normal_channel

        # Use yanx27's PointNet++ model as backbone
        # We'll modify their classification model to output embeddings
        if pnet2_version == 'pointnet2_ssg':
            self.backbone = p2_ssg_model.get_model(num_class=self.num_class, normal_channel=normal_channel)
        elif pnet2_version == 'pointnet2_msg':
            self.backbone = p2_msg_model.get_model(num_class=self.num_class, normal_channel=normal_channel)
        else:
            raise ValueError('Unknown pnet2 version: {}'.format(pnet2_version))

        # Remove the final classification layer and replace with embedding layer
        # The original model has: self.classifier = nn.Linear(1024, num_class)
        # We replace it with our MLP head
        # self.head1 = nn.Linear(1024, 512)
        # self.head2 = nn.Linear(512, 256)
        # self.head3 = nn.Linear(256, embedding_dim)

        self.head1 = nn.Linear(1024, 256)
        self.head2 = nn.Linear(256, embedding_dim)

        self.dropout = nn.Dropout(p=0.4)
        # self.bn1 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        # Optional: Add L2 normalization for better contrastive learning
        self.normalize_embeddings = normalize_embeddings

    def forward(self, xyz):
        """
        Forward pass
        Args:
            xyz: Point cloud tensor [B, N, 3] or [B, N, 6] (with normals)
        Returns:
            embeddings: [B, embedding_dim]
        """
        # PointNet++ expects input in format [B, C, N] where C is channels (3 for xyz)
        if xyz.dim() == 3:
            xyz = xyz.transpose(2, 1)  # [B, N, 3] -> [B, 3, N]

        # Get embeddings from backbone. Sould be [B, C, 1]
        _, pnet2_embeddings = self.backbone(xyz)
        assert (pnet2_embeddings.dim() == 3)
        assert (pnet2_embeddings.shape[2] == 1)
        # embeddings = self.head1(pnet2_embeddings.view(pnet2_embeddings.shape[0], pnet2_embeddings.shape[1]))
        # embeddings = F.relu(self.bn1(embeddings))
        # embeddings = F.relu(self.bn2(self.dropout(self.head2(embeddings))))
        # embeddings = self.head3(embeddings)

        embeddings = self.dropout(pnet2_embeddings.view(pnet2_embeddings.shape[0], pnet2_embeddings.shape[1]))
        embeddings = F.relu(self.bn1(embeddings))
        embeddings = F.relu(self.bn2(self.head1(embeddings)))
        embeddings = self.head2(embeddings)

        # embeddings = F.relu(self.bn2(self.dropout(self.head2(embeddings))))
        # L2 normalize embeddings for better contrastive learning
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings, embeddings, embeddings
