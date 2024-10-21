from typing import Optional


import torch
import torch.nn as nn




class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: bool = True,
        batch_average: bool = True,
        ignore_index: int = -100,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()


        self.gamma = gamma
        self.alpha = alpha
        self.batch_average = batch_average
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, size_average=size_average
        )


    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, _, _ = logit.size()


        logpt = -self.criterion(logit, target.long())
        pt = torch.exp(logpt)


        if self.alpha is not None:
            logpt *= self.alpha


        loss = -((1 - pt) ** self.gamma) * logpt


        if self.batch_average:
            loss /= n


        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F




class TMSE(nn.Module):
    """
    Temporal MSE Loss Function
    Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
    arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """


    def __init__(self, threshold: float = 4, ignore_index: int = -100) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")


    def forward(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:


        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]


            # Calculate temporal difference between adjacent frames.
            temporal_difference = F.log_softmax(pred[:, 1:], dim=1) - F.log_softmax(pred[:, :-1], dim=1)
            # Calculate squared error and clamp based on the threshold.
            loss = torch.clamp(self.mse(temporal_difference, torch.zeros_like(temporal_difference)), min=0,
                               max=self.threshold ** 2)
            # Compute the average loss for this sample.
            total_loss += torch.mean(loss)


            # loss = self.mse(
            #     F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1)
            # )
            #
            # loss = torch.clamp(loss, min=0, max=self.threshold ** 2)
            # total_loss += torch.mean(loss)


        return total_loss / batch_size




class GaussianSimilarityTMSE(nn.Module):
    """
    Temporal MSE Loss Function with Gaussian Similarity Weighting
    """


    def __init__(
        self, threshold: float = 4, sigma: float = 1.0, ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")
        self.sigma = sigma


    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: the output of model before softmax. (N, C, T)
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt, sim in zip(preds, gts, sim_index):
            valid_indices = torch.where(gt != self.ignore_index)[0]
            # pred = pred[:, torch.where(gt != self.ignore_index)[0]]
            # sim = sim[:, torch.where(gt != self.ignore_index)[0]]
            pred = pred[:, valid_indices]
            sim = sim[:, valid_indices]


            # calculate gaussian similarity
            diff = sim[:, 1:] - sim[:, :-1]
            similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * self.sigma ** 2))


            # calculate temporal mse
            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1)
            )
            loss = torch.clamp(loss, min=0, max=self.threshold ** 2)


            # gaussian similarity weighting
            loss = similarity * loss


            total_loss += torch.mean(loss)


        return total_loss / batch_size


class GaussianSmoothedSimilarity(nn.Module):
    def __init__(self,sigma=1.0):
        super(GaussianSmoothedSimilarity).__init__()
        self.sigma= sigma


    def forward(self,x:torch.Tensor):
        device = x.device


        similarity= F.cosine_similarity(x[:,:,1:], x[:,:,:-1], dim=1)
        gauss_weight= torch.exp(-torch.pow(similarity,2)/(2*self.sigma **2))
        smoothed_similarity= torch.cat([torch.ones(1,1).to(device),gauss_weight],dim=-1).unsqueeze(1)
        return  smoothed_similarity


import sys
from typing import Optional


class ActionSegmentationLoss(nn.Module):
    """
    Loss Function for Action Segmentation
    You can choose the below loss functions and combine them.
        - Cross Entropy Loss (CE)
        - Focal Loss
        - Temporal MSE (TMSE)
        - Gaussian Similarity TMSE (GSTMSE)
    """


    def __init__(
        self,
        ce: bool = True,
        focal: bool = True,
        tmse: bool = False,
        gstmse: bool = False,
        weight: Optional[float] = None,
        threshold: float = 4,
        ignore_index:int=-100,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        tmse_weight: float = 0.15,
        gstmse_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.criterions = []
        self.weights = []


        if ce:
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            )
            self.weights.append(ce_weight)


        if focal:
            self.criterions.append(FocalLoss(weight=weight, ignore_index=ignore_index))
            self.weights.append(focal_weight)


        if tmse:
            self.criterions.append(TMSE( threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)


        if gstmse:
            self.criterions.append(
                GaussianSimilarityTMSE( threshold=threshold, ignore_index=ignore_index)
            )
            self.weights.append(gstmse_weight)


        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)


    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: torch.float (N, C, T).
            gts: torch.long (N, T).
            sim_index: torch.float (N, C', T).
        """


        loss = 0.0
        for criterion, weight in zip(self.criterions, self.weights):
            if isinstance(criterion, GaussianSimilarityTMSE):
                if sim_index is None:
                    raise ValueError("sim_index must be provided for GaussianSimilarityTMSE")
                loss += weight * criterion(preds, gts.long(), sim_index)
            else:
                loss += weight * criterion(preds, gts.long())


        return loss








class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """


    def __init__(
        self,
        bce: bool = True,
        ce: bool = True,
        focal: bool = False,
        mse: bool = False,
        weight: Optional[float] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()


        self.criterions = []
        if ce:
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight)  # Use CrossEntropy for multi-class
            )


        if bce:
            self.criterions.append(
                # nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
                nn.CrossEntropyLoss(weight=weight)
            )


        if focal:
            self.criterions.append(FocalLoss())


        if mse:
            self.criterions.append(nn.MSELoss())


        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)


    def forward(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """
        loss = 0.0
        # print('pred shape gt shape ',preds.shape,gts.shape,masks.shape)
        batch_size = float(preds.shape[0])


        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):


                gt= gt.unsqueeze(0)
                pred= pred.unsqueeze(0)
                mask = mask.unsqueeze(0)  # Add extra dimension for num_classes
                # print(f"mask shape {mask.shape} , pred shape {pred.shape} , gt shape {gt.shape}")
                mask = mask.to(pred.device)
                masked_output = pred * mask  # Apply mask to output
                masked_target = gt * mask
                masked_target = masked_target.long()
                # print(f"Prediction shape (after masking): {masked_output.shape}")
                # print(f"Ground truth shape (after masking): {masked_target.shape}")


                # Compute the loss on the masked values
                loss += criterion(masked_output, masked_target)


            # print(f"Prediction shape (after masking): {pred[mask].shape}")
                # print(f"Ground truth shape (after masking): {gt[mask].shape}")
                #
                # loss += criterion(pred[mask], gt[mask])


        return loss / batch_size
