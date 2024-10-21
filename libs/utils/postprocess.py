import torch
import numpy as np
from .transforms import GaussianSmoothing
from .metrics import argrelmax
import  matplotlib.pyplot as plt

class PostProcessor(object):
    def __init__(
        self,
        name: str,
        boundary_th: int = 0.7,
        theta_t: int = 15,
        kernel_size: int = 15,
    ) -> None:
        self.func = {
            "refinement_with_boundary": self._refinement_with_boundary,
            "relabeling": self._relabeling,
            "smoothing": self._smoothing,
        }
        assert name in self.func


        self.name = name
        self.boundary_th = boundary_th
        self.theta_t = theta_t
        self.kernel_size = kernel_size


        if name == "smoothing":
            self.filter = GaussianSmoothing(self.kernel_size)


    def _is_probability(self, x: np.ndarray) -> bool:
        assert x.ndim == 3


        if x.shape[1] == 1:
            # sigmoid
            if x.min() >= 0 and x.max() <= 1:
                return True
            else:
                return False
        else:
            # softmax
            _sum = np.sum(x, axis=1).astype(np.float32)
            _ones = np.ones_like(_sum, dtype=np.float32)
            return np.allclose(_sum, _ones)


    def _convert2probability(self, x: np.ndarray) -> np.ndarray:
        """
        Args: x (N, C, T)
        """
        assert x.ndim == 3


        if self._is_probability(x):
            return x
        else:
            if x.shape[1] == 1:
                # sigmoid
                prob = 1 / (1 + np.exp(-x))
            else:
                # softmax
                prob = np.exp(x) / np.sum(np.exp(x), axis=1)
            return prob.astype(np.float32)


    def _convert2label(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 or x.ndim == 3


        if x.ndim == 2:
            return x.astype(np.int64)
        else:
            if not self._is_probability(x):
                x = self._convert2probability(x)


            label = np.argmax(x, axis=1)
            return label.astype(np.int64)


    def _refinement_with_boundary(
        self,
        outputs: np.array,
        boundaries: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Get segments which is defined as the span b/w two boundaries,
        and decide their classes by majority vote.
        Args:
            outputs: numpy array. shape (N, C, T)
                the model output for frame-level class prediction.
            boundaries: numpy array.  shape (N, 1, T)
                boundary prediction.
            masks: np.array. np.bool. shape (N, 1, T)
                valid length for each video
        Return:
            preds: np.array. shape (N, T)
                final class prediction considering boundaries.
        """


        preds = self._convert2label(outputs)
        boundaries= self._convert2label(boundaries)
        # boundaries = self._convert2probability(boundaries)


        for i, (output, pred, boundary, mask) in enumerate(
            zip(outputs, preds, boundaries, masks)
        ):
            boundary = boundary[:, mask]  # Multi-class boundary, selecting valid frames
            pred = pred[mask]
            T = pred.shape[0]
            # argrelmax applied to each class to find boundaries for each class
            boundary_idx = np.array([
                argrelmax(boundary[j], threshold=self.boundary_th)[0]
                for j in range(boundary.shape[0])
            ]).flatten()

            # Sort and add the index of the last action ending
            boundary_idx = np.sort(np.unique(np.concatenate([boundary_idx, [T]])))

            # Majority vote based refinement
            for j in range(len(boundary_idx) - 1):
                count = np.bincount(pred[boundary_idx[j]:boundary_idx[j + 1]])
                modes = np.where(count == count.max())[0]

                if len(modes) == 1:
                    mode = modes[0]
                else:
                    # Resolve tie by highest probability sum
                    prob_sum_max = 0
                    for m in modes:
                        prob_sum = output[m, boundary_idx[j]:boundary_idx[j + 1]].sum()
                        if prob_sum > prob_sum_max:
                            mode = m
                            prob_sum_max = prob_sum

                preds[i, boundary_idx[j]:boundary_idx[j + 1]] = mode

        return preds
            # 
        #     boundary = boundary[mask]
        #     idx = argrelmax(boundary, threshold=self.boundary_th)


        #     # add the index of the last action ending
        #     T = pred.shape[0]
        #     idx.append(T)


        #     # majority vote
        #     for j in range(len(idx) - 1):
        #         count = np.bincount(pred[idx[j] : idx[j + 1]])
        #         modes = np.where(count == count.max())[0]
        #         if len(modes) == 1:
        #             mode = modes
        #         else:
        #             if outputs.ndim == 3:
        #                 # if more than one majority class exist
        #                 prob_sum_max = 0
        #                 for m in modes:
        #                     prob_sum = output[m, idx[j] : idx[j + 1]].sum()
        #                     if prob_sum_max < prob_sum:
        #                         mode = m
        #                         prob_sum_max = prob_sum
        #             else:
        #                 # decide first mode when more than one majority class
        #                 # have the same number during oracle experiment
        #                 mode = modes[0]


        #         preds[i, idx[j] : idx[j + 1]] = mode


        # return preds


    def _relabeling(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Relabeling small action segments with their previous action segment
        Args:
            output: the results of action segmentation. (N, T) or (N, C, T)
            theta_t: the threshold of the size of action segments.
        Return:
            relabeled output. (N, T)
        """


        preds = self._convert2label(outputs)


        for i in range(preds.shape[0]):
            # shape (T,)
            last = preds[i][0]
            cnt = 1
            for j in range(1, preds.shape[1]):
                if last == preds[i][j]:
                    cnt += 1
                else:
                    if cnt > self.theta_t:
                        cnt = 1
                        last = preds[i][j]
                    else:
                        preds[i][j - cnt : j] = preds[i][j - cnt - 1]
                        cnt = 1
                        last = preds[i][j]


            if cnt <= self.theta_t:
                preds[i][j - cnt : j] = preds[i][j - cnt - 1]


        return preds


    def _smoothing(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Smoothing action probabilities with gaussian filter.
        Args:
            outputs: frame-wise action probabilities. (N, C, T)
        Return:
            predictions: final prediction. (N, T)
        """


        outputs = self._convert2probability(outputs)
        outputs = self.filter(torch.Tensor(outputs)).numpy()


        preds = self._convert2label(outputs)
        return preds


    def __call__(self, outputs, **kwargs: np.ndarray) -> np.ndarray:
        preds = self.func[self.name](outputs, **kwargs)
        return preds
    
    
    
    def segment_bars_with_confidence(self,save_path, confidence, *labels):
        num_pics = len(labels) + 1
        color_map = plt.get_cmap('seismic')
    
        axprops = dict(xticks=[], yticks=[], frameon=False)
        barprops = dict(aspect='auto', cmap=color_map,
                        interpolation='nearest', vmin=0)
        fig = plt.figure(figsize=(15, num_pics * 1.5))
    
        interval = 1 / (num_pics+1)
        for i, label in enumerate(labels):
            i = i + 1
            ax1 = fig.add_axes([0, 1-i*interval, 1, interval])
            ax1.imshow([label], **barprops)
    
        ax4 = fig.add_axes([0, interval, 1, interval])
        ax4.set_xlim(0, len(confidence))
        ax4.set_ylim(0, 1)
        ax4.plot(range(len(confidence)), confidence)
        ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')
    
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
    
        plt.close()

    def _plot_stage_refinement(self, preds, stage, save_dir):
        """ Plot and save the refinement at each stage. """
        # You can use the segment_bars_with_confidence or create another plot function
        fig = plt.figure(figsize=(15, 3))
        plt.title(f"Stage {stage + 1} Refinement")
        plt.imshow([preds], aspect='auto', cmap='seismic', interpolation='nearest')
        plt.colorbar()

        if save_dir:
            plt.savefig(f"{save_dir}/stage_{stage + 1}_refinement.png")
        else:
            plt.show()

        plt.close()
    def refine_segments(self,frame_predictions,boundary_predictions):
        segments= []
        current_segment= None
        for i , (action,boundary) in enumerate(zip(frame_predictions,boundary_predictions)):
            if boundary == 'start':
                if current_segment is not None:
                    segments.append(current_segment)
                current_segment = {'action': action, 'start': i}
            elif boundary == 'continue':
                if current_segment is not None:
                    current_segment