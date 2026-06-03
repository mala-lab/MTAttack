import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    def __init__(self, num_classes=3, feat_dim=2, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = torch.device(device)

        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim, device=self.device)
        )


    def forward(self, embds, labels):
        """
        Args:
            embds: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = embds.size(0)
        
        """
        Part 1: compute the squared distance between each sample and each class center.
        ||x - c||^2
           = (x - c)^T(x - c)
           = ||x||^2 + ||c||^2 - 2x^Tc

        The resulting distmat has shape [batch_size, num_classes], where
        distmat[i, j] is the squared distance between sample i and center j.
        """
        distmat = torch.pow(embds, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # print("embds.shape", embds.shape)
        # print("self.centers.t().shape", self.centers.t().shape)

        distmat.addmm_(mat1=embds, mat2=self.centers.t(), beta=1, alpha=-2)

        """
        Each sample belongs to exactly one class, so the loss only keeps the
        distance to the corresponding class center. A mask is used to filter
        out distances to the other centers.
        """
        classes = torch.arange(self.num_classes, device=embds.device).long()
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        dist_to_center = distmat * mask.float()
        loss_tpa = dist_to_center.clamp(min=1e-12, max=1e+12).sum() / batch_size

        """
        Part 2: compute cross-entropy using cosine similarity between samples
        and class centers as logits.
        """
        
        def cosine_similarity(batch_sample, class_center):
            """
            Compute cosine similarity between a batch of samples and all class centers.

            batch_sample: [batch_size, feature_dim]
            class_center: [num_classes, feature_dim]

            Returns a [batch_size, num_classes] similarity matrix.
            """
            batch_sample_norm = F.normalize(batch_sample, p=2, dim=1)
            class_center_norm = F.normalize(class_center, p=2, dim=1)
            
            similarity = torch.mm(batch_sample_norm, class_center_norm.t())
        
            return similarity
        
        logits = cosine_similarity(embds, self.centers)
        
        logits = logits.float()
        labels = labels.long()
        loss_psp = F.cross_entropy(logits, labels)

        return loss_tpa, loss_psp, logits
