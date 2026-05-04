import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extractors.encoders_pointnet import PointNetEncoder, PointNet2Encoder
# from feature_extractors.encoders_pct import PCTEncoder
from feature_extractors.encoders_riconv import RISurConvEncoder

class RepresentationHead(nn.Module):
    """A base class for representation heads"""
    def __init__(self, embedding_dim: int, normalize_embeddings: bool = True):
        super(RepresentationHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings


class RepresentationHead_512_256(RepresentationHead):
    """A standard representation head with 2 hidden layers size of 512 and 256"""
    def __init__(self, embedding_dim: int, normalize_embeddings: bool = True, use_dropout=True):
        super(RepresentationHead_512_256, self).__init__(embedding_dim, normalize_embeddings)
        self.head1 = nn.Linear(512, 256)
        self.head1.gradnorm = True
        self.head2 = nn.Linear(256, embedding_dim)
        self.head2.gradnorm = True
        self.dropout = nn.Dropout(p=0.4)
        self.use_dropout = use_dropout
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, logits):
        assert (logits.dim() == 3)
        assert (logits.shape[2] == 1)
        embeddings = logits.view(logits.shape[0], logits.shape[1])
        if self.use_dropout:
            embeddings = self.dropout(embeddings)
        embeddings = F.relu(self.bn1(embeddings))
        embeddings = F.relu(self.bn2(self.head1(embeddings)))
        embeddings = self.head2(embeddings)
        if self.normalize_embeddings:
            # L2 normalize embeddings for better contrastive learning
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class MultiTaskMetricHead2(nn.Module):
    """This is a base head for multi task learning"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        # Note: currently we can't use GradNorm in the MTL heads.
        # For a head, only one loss is defined; but our GradNorm code can't handle it yet.
        self.head_mined1  = nn.Linear(embedding_dim, embedding_dim*2)
        self.head_mined2 = nn.Linear(embedding_dim*2, embedding_dim)
        self.head_forced1  = nn.Linear(embedding_dim, embedding_dim*2)
        self.head_forced2 = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, x):
        # Should be NO L2-normalization here
        features = self.features_backbone(x)
        z_mined  = self.head_mined2(F.relu(self.head_mined1(features)))
        z_forced = self.head_forced2(F.relu(self.head_forced1(features)))
        # Do embedding normalization
        z_mined  = nn.functional.normalize(z_mined,  p=2, dim=1)
        z_forced = nn.functional.normalize(z_forced, p=2, dim=1)
        return z_mined, z_forced

class MultiTaskMetricHead1(nn.Module):
    """This is a base head for multi task learning"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.head_mined1  = nn.Linear(embedding_dim, embedding_dim)
        self.head_forced1  = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings):
        # Should be NO L2-normalization here
        z_mined  = self.head_mined1(embeddings)
        z_forced = self.head_forced1(embeddings)
        # Do embedding normalization
        z_mined  = nn.functional.normalize(z_mined,  p=2, dim=1)
        z_forced = nn.functional.normalize(z_forced, p=2, dim=1)
        return z_mined, z_forced

def instantiate_model(model_name:str, embedding_dim:int, normalize_embeddings:bool=True, use_dropout:bool=True):
    # Create model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'pointnet':
        model = PointNetEncoder(embedding_dim=embedding_dim)
    elif model_name == 'pointnet2_ssg' or model_name == 'pointnet2_msg':
        model = PointNet2Encoder(embedding_dim=embedding_dim)
    # elif model_name == 'pct':
    #     model = PCTEncoder(embedding_dim=embedding_dim)
    # elif model_name == 'pct_compact':
    #     model = PCTEncoder(embedding_dim=embedding_dim)
    elif model_name == 'riconv2':
        rep_head = RepresentationHead_512_256(
            embedding_dim=embedding_dim, normalize_embeddings=normalize_embeddings, use_dropout=use_dropout)
        # Embeddings should be already normalized by the representation head, no need to do it in the full model
        model = RISurConvEncoder(rep_head=rep_head, normalize_embeddings=False)
    elif model_name == 'riconv2_mtl1':
        # For MTL mode, we MUST NOT normalize embeddings in the head
        rep_head = RepresentationHead_512_256(
            embedding_dim=embedding_dim, normalize_embeddings=False, use_dropout=use_dropout)
        mtl_head = MultiTaskMetricHead1(embedding_dim=embedding_dim)
        # If requested, do normalization in the model itself
        model = RISurConvEncoder(rep_head=rep_head, mtl_head=mtl_head, normalize_embeddings=normalize_embeddings)
    elif model_name == 'riconv2_mtl2':
        rep_head = RepresentationHead_512_256(
            embedding_dim=embedding_dim, normalize_embeddings=False, use_dropout=use_dropout)
        mtl_head = MultiTaskMetricHead2(embedding_dim=embedding_dim)
        model = RISurConvEncoder(rep_head=rep_head, mtl_head=mtl_head, normalize_embeddings=normalize_embeddings)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model
