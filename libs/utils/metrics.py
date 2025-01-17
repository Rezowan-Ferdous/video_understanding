import numpy as np
import pandas as pd
import copy
import csv
from typing import Dict,List,Optional,Tuple
import torch
# from .transforms import GaussianSmoothing


def get_segments(
        frame_wise_label:np.ndarray,
        bg_class=[-100]
):
    # frame_wise_label= [frame_wise_label[i] for i in range(len(frame_wise_label))]
    labels=[]
    starts=[]
    ends=[]


    last_label = frame_wise_label[0]
    labels.append(frame_wise_label[0])
    starts.append(0)


    for i in range(1, len(frame_wise_label)):
        if frame_wise_label[i] != last_label:
            if last_label not in bg_class:
                ends.append(i)  # Capture the end of the previous segment
            if frame_wise_label[i] not in bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)
            last_label = frame_wise_label[i]


    # Capture the last segment
    if last_label not in bg_class:
        ends.append(len(frame_wise_label))


    return labels, starts, ends




def levenshtein(pred: List[int], gt: List[int], norm: bool = True) -> float:
    """
    Levenshtein distance(Edit Distance)
    Args:
        pred: segments list
        gt: segments list
    Return:
        if norm == True:
            (1 - average_edit_distance) * 100
        else:
            edit distance
    """


    n, m = len(pred), len(gt)


    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j


    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + cost,
            )  # replacement


    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100
    else:
        score = dp[n][m]


    return score




def func_eval(gt_content, recog_content):
    # ground_truth_path = "./data/" + dataset + "/groundTruth/"
    # mapping_file = "./data/" + dataset + "/mapping.txt"
    # list_of_videos = read_file(file_list).split('\n')[:-1]


    # file_ptr = open(mapping_file, 'r')
    # actions = file_ptr.read().split('\n')[:-1]
    # file_ptr.close()
    # actions_dict = dict()
    # for a in actions:
    #     actions_dict[a.split()[1]] = int(a.split()[0])


    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)


    correct = 0
    total = 0
    edit = 0


    # for vid in list_of_videos:
    #
    #     gt_file = ground_truth_path + vid
    #     gt_content = read_file(gt_file).split('\n')[0:-1]
    #
    #     recog_file = recog_path + vid.split('.')[0]
    #     recog_content = read_file(recog_file).split('\n')[1].split()


    for i in range(len(gt_content)):
        total += 1
        if gt_content[i] == recog_content[i]:
            correct += 1


    edit += edit_score(recog_content, gt_content)


    for s in range(len(overlap)):
        tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
        tp[s] += tp1
        fp[s] += fp1
        fn[s] += fn1


    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / 1
    #     print("Acc: %.4f" % (acc))
    #     print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0, 0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])


    f1 = 2.0 * (precision * recall) / (precision + recall)


    f1 = np.nan_to_num(f1) * 100
    #         print('F1@%0.2f: %.4f' % (overlap[s], f1))
    f1s[s] = f1


    return acc, edit, f1s




def get_n_samples(
    p_label: List[int],
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
    iou_threshold: float,
    soft_matching: bool = True,
    bg_class: List[int] = [-100],
) -> Tuple[int, int, int]:
    """
    Args:
        p_label, p_start, p_end: return values of get_segments(pred)
        g_label, g_start, g_end: return values of get_segments(gt)
        threshold: threshold (0.1, 0.25, 0.5)
        bg_class: background class
    Return:
        tp: true positive
        fp: false positve
        fn: false negative
    """
    # Edge Case: No predictions
    if len(p_label) == 0:
        return 0, 0, len(g_label)


    # Edge Case: No ground truth segments
    if len(g_label) == 0:
        return 0, len(p_label), 0


    tp = 0
    fp = 0
    hits = np.zeros(len(g_label))


    for j in range(len(p_label)):
        # Compute intersection (handles broadcasting between scalar and array)
        intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)


        # Clip negative intersections (if any)
        valid_intersection = np.maximum(0, intersection)


        # Since p_end[j] is a scalar, don't try to print its shape
        # print(f"p_end[{j}]: {p_end[j]}, no shape (scalar)")
        # print(f"g_end: {g_end}, shape: {np.array(g_end).shape}")
        #
        # print(f"Intersection: {valid_intersection}")
        intersection=valid_intersection


        # intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)




        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == g_label[x] for x in range(len(g_label))]
        )
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        best_IoU = IoU[idx]
        # Handle soft-matching: allow IoU contribution below threshold
        if soft_matching:
            if best_IoU > 0 and not hits[idx]:
                tp += best_IoU  # Fractional match based on IoU
                hits[idx] = 1  # Mark this ground truth segment as matched
            else:
                fp += (1 - best_IoU)  # Add as fractional false positive if IoU is low
        else:
            # Strict IoU matching based on threshold
            if best_IoU >= iou_threshold and not hits[idx]:
                tp += 1
                hits[idx] = 1  # Mark the ground truth as matched
            else:
                fp += 1  # Full false positive if IoU is below threshold or already matched


        # if IoU[idx] >= iou_threshold and not hits[idx]:
        #     tp += 1
        #     hits[idx] = 1
        # else:
        #     fp += 1
    #
    # fn = len(g_label) - sum(hits)




    fn = len(g_label) - sum(hits)


    return int(tp), int(fp), int(fn)




def edit_score(recognized, ground_truth, norm=True, bg_class=[-100]):
    P, _, _ = get_segments(recognized)
    Y, _, _ = get_segments(ground_truth)
    return levenshtein(P, Y, norm)




def f_score(recognized, ground_truth, overlap, bg_class=[-100]):
    p_label, p_start, p_end = get_segments(recognized)
    y_label, y_start, y_end = get_segments(ground_truth)


    tp = 0
    fp = 0


    hits = np.zeros(len(y_label))


    for j in range(len(p_label)):
        # Ensure p_start[j] and p_end[j] are scalars, and g_start, g_end are lists/arrays
        # print(f"Predicted: p_start[{j}] = {p_start[j]}, p_end[{j}] = {p_end[j]}")
        # print(f"Ground truth: g_start = {y_start}, g_end = {y_end}")


        # Compute intersection (handles broadcasting between scalar and array)
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)


        # Clip negative intersections (if any)
        valid_intersection = np.maximum(0, intersection)


        # Since p_end[j] is a scalar, don't try to print its shape
        # print(f"p_end[{j}]: {p_end[j]}, no shape (scalar)")
        # print(f"g_end: {y_end}, shape: {np.array(y_end).shape}")
        #
        # print(f"Intersection: {valid_intersection}")
        intersection = valid_intersection
        # intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()


        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


class ScoreMeter(object):
    def __init__(
        self,
        id2class_map: Dict[int, str],
        iou_thresholds: Tuple[float] = (0.1, 0.25, 0.5),
        ignore_indexes: Tuple[int] = (255,-100,-1),
    ) -> None:


        self.iou_thresholds = iou_thresholds  # threshold for f score
        self.ignore_index = ignore_indexes
        self.id2class_map = id2class_map
        self.edit_score = 0
        self.tp = [0 for _ in range(len(iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.n_classes = len(self.id2class_map)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


    def _fast_hist(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        mask = (gt >= 0) & (gt < self.n_classes)
        hist = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist


    def update(
        self,
        outputs: np.ndarray,
        gts: np.ndarray,
        boundaries: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            outputs: np.array. shape(N, C, T)
                the model output for boundary prediciton
            gt: np.array. shape(N, T)
                Ground Truth for boundary
        """
        if len(outputs.shape) == 3:
            preds = outputs.argmax(axis=1)
        elif len(outputs.shape) == 2:
            preds = copy.copy(outputs)


        for pred, gt in zip(preds, gts):


            # pred = pred[~np.isin(gt, self.ignore_index)]
            # gt = gt[~np.isin(gt, self.ignore_index)]
            # pred = pred[gt not in self.ignore_index]
            # gt = gt[gt not in  self.ignore_index]


            if isinstance(gt, np.ndarray):  # If gt is a numpy array
                pred = pred[~np.isin(gt, self.ignore_index)]
            else:  # If gt is a list
                pred = [p for p, g in zip(pred, gt) if g not in self.ignore_index]


            for lt, lp in zip(pred, gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())


            self.n_videos += 1
            # count the correct frame
            self.n_frames += len(pred)
            for i in range(len(pred)):
                if pred[i] == gt[i]:
                    self.n_correct += 1


            # acc,edit,f1= func_eval(gt_content=gt,recog_content=pred)
            # self.edit_score= edit


            # calculate the edit distance
            p_label, p_start, p_end = get_segments(pred)
            g_label, g_start, g_end = get_segments(gt)


            self.edit_score += levenshtein(p_label, g_label, norm=True)


            for i, th in enumerate(self.iou_thresholds):
                tp, fp, fn = get_n_samples(
                    p_label, p_start, p_end, g_label, g_start, g_end, th
                )
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn


    def get_scores(self) -> Tuple[float, float, float]:
        """
        Return:
            Accuracy
            Normlized Edit Distance
            F1 Score of Each Threshold
        """


        # accuracy
        acc = 100 * float(self.n_correct) / self.n_frames


        # edit distance
        edit_score = float(self.edit_score) / self.n_videos


        # F1 Score
        f1s = []
        for i in range(len(self.iou_thresholds)):
            # precision = self.tp[i] / float(self.tp[i] + self.fp[i])
            # recall = self.tp[i] / float(self.tp[i] + self.fn[i])
            #
            # f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
            # f1 = np.nan_to_num(f1) * 100
            #
            # f1s.append(f1)


            # for i in range(len(self.iou_thresholds)):
            # Check for zero denominator
            if self.tp[i] + self.fp[i] == 0:
                precision = 0  # Handle case when no positives
            else:
                precision = self.tp[i] / float(self.tp[i] + self.fp[i])


            if self.tp[i] + self.fn[i] == 0:
                recall = 0  # Handle case when no true labels
            else:
                recall = self.tp[i] / float(self.tp[i] + self.fn[i])


            if precision + recall == 0:
                f1 = 0  # Avoid NaN when both precision and recall are zero
            else:
                f1 = 2.0 * (precision * recall) / (precision + recall)


            f1 = np.nan_to_num(f1) * 100
            f1s.append(f1)


        # Accuracy, Edit Distance, F1 Score
        return acc, edit_score, f1s


    def return_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix


    def save_scores(self, save_path: str) -> None:
        acc, edit_score, segment_f1s = self.get_scores()


        # save log
        columns = ["cls_acc", "edit"]
        data_dict = {
            "cls_acc": [acc],
            "edit": [edit_score],
        }


        for i in range(len(self.iou_thresholds)):
            key = "segment f1s@{}".format(self.iou_thresholds[i])
            columns.append(key)
            data_dict[key] = [segment_f1s[i]]


        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)


    def save_confusion_matrix(self, save_path: str) -> None:
        with open(save_path, "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(self.confusion_matrix)


    def reset(self) -> None:
        self.edit_score = 0
        self.tp = [0 for _ in range(len(self.iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(self.iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(self.iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))




def argrelmax(prob: np.ndarray, threshold: float = 0.7) -> List[int]:
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold


    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0


    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )


    peak_idx = np.where(peak)[0].tolist()


    return peak_idx




class BoundaryScoreMeter(object):
    def __init__(self, tolerance=5, boundary_threshold=0.7):
        # max distance of the frame which can be regarded as correct
        self.tolerance = tolerance


        # threshold of the boundary value which can be regarded as action boundary
        self.boundary_threshold = boundary_threshold
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0


    def update(self, preds, gts, masks):
        """
        Args:
            preds: np.array. the model output(N, T)
            gts: np.array. boudnary ground truth array (N, T)
            masks: np.array. np.bool. valid length for each video (N, T)
        Return:
            Accuracy
            Boundary F1 Score
        """


        for pred, gt, mask in zip(preds, gts, masks):
            # ignore invalid frames
            pred = pred[mask]
            gt = gt[mask]


            pred_idx = argrelmax(pred, threshold=self.boundary_threshold)
            gt_idx = argrelmax(gt, threshold=self.boundary_threshold)


            n_frames = pred.shape[0]
            tp = 0.0
            fp = 0.0
            fn = 0.0


            hits = np.zeros(len(gt_idx))


            # calculate true positive, false negative, false postive, true negative
            for i in range(len(pred_idx)):
                dist = np.abs(np.array(gt_idx) - pred_idx[i])
                min_dist = np.min(dist)
                idx = np.argmin(dist)


                if min_dist <= self.tolerance and hits[idx] == 0:
                    tp += 1
                    hits[idx] = 1
                else:
                    fp += 1


            fn = len(gt_idx) - sum(hits)
            tn = n_frames - tp - fp - fn


            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.n_frames += n_frames
            self.n_correct += tp + tn


    def get_scores(self):
        """
        Return:
            Accuracy
            Boundary F1 Score
        """


        # accuracy
        acc = 100 * self.n_correct / self.n_frames


        # Boudnary F1 Score
        precision = self.tp / float(self.tp + self.fp)
        recall = self.tp / float(self.tp + self.fn)


        f1s = 2.0 * (precision * recall) / (precision + recall + 1e-7)
        f1s = np.nan_to_num(f1s) * 100


        # Accuracy, Edit Distance, F1 Score
        return acc, precision * 100, recall * 100, f1s


    def save_scores(self, save_path: str) -> None:
        acc, precision, recall, f1s = self.get_scores()


        # save log
        columns = ["bound_acc", "precision", "recall", "bound_f1s"]
        data_dict = {
            "bound_acc": [acc],
            "precision": [precision],
            "recall": [recall],
            "bound_f1s": [f1s],
        }


        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)


    def reset(self):
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0




class AverageMeter(object):
    """Computes and stores the average and current value"""


    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()


    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)




import torch.nn as nn
import torch.optim as optim




def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:


    assert optimizer_name in ["SGD", "Adam"]
    print(f"{optimizer_name} will be used as an optimizer.")


    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


    return optimizer




# __all__ = ["PostProcessor"]




def decide_boundary_prob_with_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Decide action boundary probabilities based on adjacent frame similarities.
    Args:
        x: frame-wise video features (N, C, T)
    Return:
        boundary: action boundary probability (N, 1, T)
    """
    device = x.device


    # gaussian kernel.
    diff = x[0, :, 1:] - x[0, :, :-1]
    similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * 1.0))


    # define action starting point as action boundary.
    start = torch.ones(1).float().to(device)
    boundary = torch.cat([start, similarity])
    boundary = boundary.view(1, 1, -1)
    return boundary


