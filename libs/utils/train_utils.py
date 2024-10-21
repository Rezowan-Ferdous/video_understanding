import os
from typing import Optional,Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .metrics import AverageMeter, BoundaryScoreMeter, ScoreMeter
from .postprocess import PostProcessor
import matplotlib.pyplot as plt




# from torch.cuda.amp import autocast, GradScaler


from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/experiment_name")


scaler = GradScaler()


import torch.distributed as dist
import torch.multiprocessing as mp


def setup_ddp(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()




def train_ef(
        train_loader: DataLoader,
        model,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        lambda_bound_loss: float,
        optimizer,
        epoch: int,
        device: str,
        accumulation_steps=4,  # For gradient accumulation
        mode="ms",test_loader=None,dataset=None):


    losses = AverageMeter("Loss", ":.4e")
    model = model.to(device)
    mse = nn.MSELoss(reduction='none')
    model.train()
    total_correct = 0  # Initialize accuracy accumulator
    total_samples = 0
    total_b_correct=0
    total_b_samples=0


    optimizer.zero_grad()  # Start by zeroing the gradients
    for i, sample in enumerate(train_loader):
        x = sample["feature"].to(device)
        t = sample["label"].to(device)
        b = sample["boundary"].to(device)
        mask = sample["mask"].to(device)


        batch_size = x.shape[0]
        # print(f" x shape {x.shape} target shape {t.shape} boundary shape {b.shape} mask shape {mask.shape}")


        # Define the number of elements to sample


        max_index = t.shape[1]
        if dataset=="cholec":
            n_elements = max_index // 5
        elif dataset=="jigsaw":
            n_elements = max_index // 2
        elif dataset =="rarp50":
            n_elements = max_index // 2
        else:
            n_elements = max_index // 2




        # print(' elements ',n_elements)


        indices = np.linspace(0, max_index - 1, n_elements).astype(int)
        #
        # # Sample the sequence, label, and mask using the generated indices
        x = x[:, :, indices]  # Select along the time dimension
        t = t[:, indices]  # Select along the time dimension
        mask = mask[:, indices]  # Select along the time dimension
        b = b[:, indices]


        # Use mixed precision autocasting
        with autocast(device_type='cuda'):
            output_cls, output_bound = model(x, mask)
            loss = 0.0
            lossb=0.0


            # Classification loss
            if mode == "ms":
                msloss=0.0
                for p in output_cls:
                    msloss += criterion_cls(p, t, x)
                # print(f'msloss {msloss}, output shape {output_cls.shape[0]}')
                msloss = msloss/output_cls.shape[0]


                loss += msloss


                msloss_b = 0.0
                for bl in output_bound:
                    bl = bl.to(device)
                    msloss_b += criterion_bound(bl, b, x)
                # print(f'msloss {msloss}, output shape {output_cls.shape[0]}')
                msloss_b = msloss_b / output_bound.shape[0]


                loss += msloss_b


            elif isinstance(output_cls, list):
                n = len(output_cls)
                for out in output_cls:
                    loss += criterion_cls(out, t, x) / n
                # n = len(output_bound)
                for out in output_bound:
                    loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
            else:
                loss += criterion_cls(output_cls, t, x)


                loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)






        # Accumulate gradients
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()  # Scaled backpropagation with AMP


        # Gradient update
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # Apply the optimizer step
            scaler.update()  # Update the scale for next iteration
            optimizer.zero_grad()  # Zero the gradients after each step








        losses.update(loss.item() * accumulation_steps, batch_size)  # Record the loss
        # correct = 0
        # total = 0
        if mode == "ms":
            _, predicted = torch.max(output_cls.data[-1], 1)
            _, predicted_b = torch.max(output_bound.data[-1], 1)
            mask = mask.unsqueeze(1).to(device)
            predicted = predicted.to(device)


            predicted_b= predicted_b.to(device)


            # correct += ((predicted == t).float() * mask[:, 0, :].squeeze(1)).sum().item()
            # total += torch.sum(mask[:, 0, :]).item()
            # Accumulate correct predictions across batches
            total_correct += ((predicted == t).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total_samples += torch.sum(mask[:, 0, :]).item()


            total_b_correct+= ((predicted_b == b).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total_b_samples += torch.sum(mask[:, 0, :]).item()




    # Print once per epoch
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"[epoch {epoch + 1}]: epoch loss = {losses.avg}, acc = {accuracy:.4f}")


    if (epoch + 1) % 5 == 0 and test_loader is not None:
        test(model, test_loader, epoch, device)
        # torch.save(model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
        # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")


    return losses.avg


def train(
        train_loader: DataLoader,
        model,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        lambda_bound_loss: float,
        optimizer,
        epoch: int,
        device: str,
        mode="ms",test_loader=None):
    losses = AverageMeter("Loss",":.4e")
    model = model.to(device)
    mse = nn.MSELoss(reduction='none')
    model.train()
    for i, sample in enumerate(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]
        x = x.to(device)
        t = t.to(device)
        b = b.to(device)


        batch_size= x.shape[0]
        # print(f" x shpe {x.shape} target shape {t.shape} boundary shape {b.shape} mask shape {mask.shape}")
        output_cls,output_bound= model(x, mask)


        loss = 0.0
        batch_target=t
        if mode=="ms":
            for p in output_cls:
                loss += criterion_cls(p, t, x)
                # loss += criterion_cls(p.transpose(2, 1).contiguous().view(-1, num_class), batch_target.view(-1),x)
                # loss += 0.15 * torch.mean(torch.clamp(
                #     mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                #     max=16) * mask[:, :, 1:])
                # epoch_loss += loss.item()




        elif isinstance(output_cls, list):
            n =len(output_cls)
            for out in output_cls:
                # print('output ', out.shape)
                loss+=criterion_cls(out,t,x)/n


        else:
            loss+=criterion_cls(output_cls,t,x)


        if isinstance(output_bound,list):
            n= len(output_bound)
            for out in output_bound:
                # print('output ',out.shape)
                loss+=lambda_bound_loss * criterion_bound(out,b,mask)/n
        else:
            loss+=lambda_bound_loss*criterion_bound(output_bound,b,mask)


        losses.update(loss.item(),batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        correct= 0
        total=0
        if mode=="ms":
            _, predicted = torch.max(output_cls.data[-1], 1)
            mask=mask.unsqueeze(1).to(device)
            predicted = predicted.to(device)


            correct += ((predicted == t).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total += torch.sum(mask[:, 0, :]).item()
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, losses.avg,
                                                               float(correct) / total))


            if (epoch + 1) % 2 == 0 and test_loader is not None:
                test(model, test_loader, epoch,device)
                # torch.save(model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")


    return losses.avg


def validate(
        val_loader:DataLoader,
        model:nn.Module,
        criterion_cls:nn.Module,
        criterion_bound:nn.Module,
        lambda_bound_loss:float,
        device:str,
        dataset: str,
        dataset_dir: str,
        action_dict,
        iou_thresholds: Tuple[float],
        boundary_th: float,
        tolerance: int,
        mode='ms',
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        losses = AverageMeter("Loss", ":.4e")
        scores_cls = ScoreMeter(
            # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
            id2class_map= action_dict,
            iou_thresholds=iou_thresholds,
        )
        scores_bound = ScoreMeter(
            id2class_map={'continue':0 , 'start':1 , 'end':2},
            iou_thresholds=iou_thresholds,
        )


        # scores_bound = BoundaryScoreMeter(
        #     tolerance=tolerance, boundary_threshold=boundary_th
        # )


        # switch to evaluate mode
        model.eval()


        with torch.no_grad():
            for sample in val_loader:
                x = sample["feature"]
                t = sample["label"]
                b = sample["boundary"]
                mask = sample["mask"]


                x = x.to(device)
                t = t.to(device)
                b = b.to(device)
                mask = mask.to(device)


                batch_size = x.shape[0]


                # compute output and loss
                output_cls, output_bound = model(x,mask)


                loss = 0.0
                # Classification loss
                if mode == "ms":
                    msloss = 0.0
                    for p in output_cls:
                        msloss += criterion_cls(p, t, x)
                    # print(f'msloss {msloss}, output shape {output_cls.shape[0]}')
                    msloss = msloss / output_cls.shape[0]


                    loss += msloss


                    msloss_b = 0.0
                    for bl in output_bound:
                        bl = bl.to(device)
                        msloss_b += criterion_bound(bl, b, x)
                    # print(f'msloss {msloss}, output shape {output_cls.shape[0]}')
                    msloss_b = msloss_b / output_bound.shape[0]


                    loss += msloss_b


                elif isinstance(output_cls, list):
                    n = len(output_cls)
                    for out in output_cls:
                        loss += criterion_cls(out, t, x) / n
                    # n = len(output_bound)
                    for out in output_bound:
                        loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
                else:
                    loss += criterion_cls(output_cls, t, x)


                    # loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)
                    loss+= criterion_bound(output_bound, b, mask)


                # # Accumulate gradients
                # loss = loss / accumulation_steps
                # scaler.scale(loss).backward()  # Scaled backpropagation with AMP


                # measure accuracy and record loss
                losses.update(loss.item(), batch_size)


                # calcualte accuracy and f1 score
                output_cls = output_cls.to("cpu").data.numpy()
                output_bound = output_bound.to("cpu").data.numpy()


                t = t.to("cpu").data.numpy()
                b = b.to("cpu").data.numpy()
                mask = mask.to("cpu").data.numpy()


                # update score
                out_class= torch.mean(torch.from_numpy(output_cls), dim=0).numpy()
                bound_class = torch.mean(torch.from_numpy(output_bound), dim=0).numpy()


                scores_cls.update(out_class, t, bound_class, mask)
                scores_bound.update(bound_class, b)


        cls_acc, edit_score, segment_f1s = scores_cls.get_scores()
        # bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()
        bound_acc, bound_edit, bound_f1s = scores_bound.get_scores()


        return (
            losses.avg,
            cls_acc,
            edit_score,
            segment_f1s,
            bound_acc,
            bound_edit,
            bound_f1s,
        )




def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    action_dict,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)


    scores_before_refinement = ScoreMeter(
        # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        id2class_map= action_dict,
        iou_thresholds=iou_thresholds,
    )
    scores_bound= ScoreMeter(
        # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        id2class_map={'continue':0, 'start':1,'end':2},
        iou_thresholds=iou_thresholds,
    )


    # scores_bound = BoundaryScoreMeter(
    #     tolerance=tolerance, boundary_threshold=boundary_th
    # )


    scores_after_refinement = ScoreMeter(
        # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        id2class_map=action_dict,
        iou_thresholds=iou_thresholds,
    )


    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]


            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)


            # compute output and loss
            output_cls, output_bound = model(x)


            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()


            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()


            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )


            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)


    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())


    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )


    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))


    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )


import numpy as np




def test(model, test_loader, epoch,device):
    model.eval()
    correct = 0
    total = 0
    if_warp = False  # When testing, always false
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]
            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask=mask.to(device)


            p,pb = model(x, mask)
            _, predicted = torch.max(p[-1], 1)
            # correct += ((predicted == t).float() * mask[:, 0, :].squeeze(1)).sum().item()
            # total += torch.sum(mask[:, 0, :]).item()


            # Adjust indexing based on mask's shape
            if len(mask.shape) == 2:
                correct += ((predicted == t).float() * mask).sum().item()  # Adjust for 2D mask
                total += torch.sum(mask[:, :]).item()
            elif len(mask.shape) == 3:
                correct += ((predicted == t).float() * mask[:, 0, :].squeeze(
                    1)).sum().item()  # Original indexing for 3D mask
                total += torch.sum(mask[:, 0, :]).item()


    acc = float(correct) / total
    print("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc))


    model.train()




def predict(
    loader: DataLoader,
    model: nn.Module,
    device: str,
    result_path: str,
    boundary_th: float,
) -> None:
    save_dir = os.path.join(result_path, "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    postprocessor = PostProcessor("refinement_with_boundary", boundary_th)


    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in loader:
            x = sample["feature"]
            t = sample["label"]
            path = sample["feature_path"][0]
            name = os.path.basename(path)
            mask = sample["mask"].numpy()


            x = x.to(device)


            # compute output and loss
            output_cls, output_bound = model(x)


            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()


            refined_pred = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )


            pred = output_cls.argmax(axis=1)


            np.save(os.path.join(save_dir, name[:-4] + "_pred.npy"), pred[0])
            np.save(
                os.path.join(save_dir, name[:-4] + "_refined_pred.npy"), refined_pred[0]
            )
            np.save(os.path.join(save_dir, name[:-4] + "_gt.npy"), t[0])


            # make graph for boundary regression
            output_bound = output_bound[0, 0]
            h_axis = np.arange(len(output_bound))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.tick_params(labelbottom=False, labelright=False, labeltop=False)
            plt.ylim(0.0, 1.0)
            ax.set_yticks([0, boundary_th, 1])
            ax.spines["right"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.plot(h_axis, output_bound, color="#e46409")
            plt.savefig(os.path.join(save_dir, name[:-4] + "_boundary.png"))
            plt.close(fig)


def save_checkpoint(
        result_path:str,
        epoch:int,
        model:nn.Module,
        optimizer:optim.Optimizer,
        best_loss:float,
):
    save_states={
        "epoch":epoch,
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        best_loss:best_loss,
    }
    torch.save(save_states,os.path.join(result_path,"checkpoint.pth"))


def resume(
        result_path:str,
        model:nn.Module,
        optimizer:optim.Optimizer,
):
    resume_path=os.path.join(result_path,"checkpoint.pth")
    print("loading checkpoint {}".format(result_path))


    checkpoint = torch.load(resume_path, map_location=lambda storage,loc:storage)
    begin_epoch= checkpoint["epoch"]
    best_loss= checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])


    optimizer.load_state_dict(checkpoint["optimizer"])


    return begin_epoch, model,optimizer,best_loss


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


# ===== class weight =======


def get_class_weight(num_class,dataframe):
    nums = [0 for i in range(num_class)]
    bounds = [0 for i in range(3)]
    for idx, row in dataframe.iterrows():
        labels = os.path.join(os.path.dirname(row['feature_path']), 'labels.npy')
        boundaries = os.path.join(os.path.dirname(row['feature_path']), 'boundaries.npy')
        label = np.load(labels)
        boundary = np.load(boundaries)
        num, cnt = np.unique(label, return_counts=True)
        b, ct = np.unique(boundary, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c
        for i, j in zip(b, ct):
            # print(i,j)
            bounds[i] += j


    print(nums, bounds)


    class_num = torch.tensor(nums)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency
    pos_num = torch.tensor(bounds)
    totalb = pos_num.sum().item()
    frequencyb = pos_num.float() / totalb
    medianb = torch.median(frequencyb)
    pos_weight = medianb / frequencyb
    print(class_weight,pos_weight)
    return class_weight,pos_weight


import json
def get_class_weight_u(num_class, dataframe):
    nums = [0 for _ in range(num_class)]
    bounds = [0 for _ in range(3)]


    for idx, row in dataframe.iterrows():
        # Handle both file paths and numpy arrays
        labels_path = os.path.join(os.path.dirname(row['feature_path']), 'labels.npy')
        boundaries_path = os.path.join(os.path.dirname(row['feature_path']), 'boundaries.npy')


        # Check if labels and boundaries are file paths or numpy arrays
        if isinstance(row['labels'], str) and row['labels'].endswith('.npy'):
            label = np.load(labels_path)
        else:
            label = row['labels']


        if isinstance(row['boundaries'], str) and row['boundaries'].endswith('.npy'):
            boundary = np.load(boundaries_path)
        else:
            boundary = row['boundaries']


        # Ignore -100 and 255 in label and boundary arrays
        valid_label_mask = (label != -100) & (label != 255)
        valid_boundary_mask = (boundary != -100) & (boundary != 255)


        label = label[valid_label_mask]
        boundary = boundary[valid_boundary_mask]


        # Count label occurrences
        num, cnt = np.unique(label, return_counts=True)
        b, ct = np.unique(boundary, return_counts=True)


        for n, c in zip(num, cnt):
            nums[n] += c
        for i, j in zip(b, ct):
            bounds[i] += j




    # Calculate class weights
    class_num = torch.tensor(nums)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency


    # Calculate boundary weights
    pos_num = torch.tensor(bounds)
    totalb = pos_num.sum().item()
    frequencyb = pos_num.float() / totalb
    medianb = torch.median(frequencyb)
    pos_weight = medianb / frequencyb


    print(class_weight, pos_weight)
    return class_weight, pos_weight




def get_class_weight_cholec(num_class, dataframe, max_weight=1000):
    nums = [0 for _ in range(num_class)]
    bounds = [0 for _ in range(3)]


    for idx, row in dataframe.iterrows():
        labels_path = row['labels']
        boundaries_path = row['boundaries']
        label = np.load(labels_path)
        boundary = np.load(boundaries_path)


        # Ignore -100 and 255 in label and boundary arrays
        valid_label_mask = (label != -100) & (label != 255)
        valid_boundary_mask = (boundary != -100) & (boundary != 255)


        label = label[valid_label_mask]
        boundary = boundary[valid_boundary_mask]


        # Count label occurrences
        num, cnt = np.unique(label, return_counts=True)
        b, ct = np.unique(boundary, return_counts=True)


        for n, c in zip(num, cnt):
            nums[n] += c
        for i, j in zip(b, ct):
            bounds[int(i)] += j


    # Calculate class weights
    class_num = torch.tensor(nums)
    total = class_num.sum().item()


    # Handle cases where total might be zero to avoid division by zero
    if total == 0:
        raise ValueError("Total count of labels is zero. Please check the input data.")


    frequency = class_num.float() / total
    median = torch.median(frequency)


    # Replace zeros in frequency to avoid infinite class weights
    frequency[frequency == 0] = float('inf')


    class_weight = median / frequency


    # Cap class weights to avoid extremely large values
    class_weight = torch.clamp(class_weight, max=max_weight)


    # Calculate boundary weights
    pos_num = torch.tensor(bounds)
    totalb = pos_num.sum().item()


    # Handle cases where totalb might be zero to avoid division by zero
    if totalb == 0:
        raise ValueError("Total count of boundaries is zero. Please check the input data.")


    frequencyb = pos_num.float() / totalb
    medianb = torch.median(frequencyb)


    # Replace zeros in boundary frequency to avoid infinite boundary weights
    frequencyb[frequencyb == 0] = float('inf')


    pos_weight = medianb / frequencyb


    # Cap boundary weights to avoid extremely large values
    pos_weight = torch.clamp(pos_weight, max=max_weight)


    print(class_weight, pos_weight)
    return class_weight, pos_weight




# nums = [0 for i in range(8)]
# bounds = [0 for i in range(3)]
# print(nums, bounds)
# for idx, row in dataframe.iterrows():
#     labels = os.path.join(os.path.dirname(row['feature_path']), 'labels.npy')
#     boundaries = os.path.join(os.path.dirname(row['feature_path']), 'boundaries.npy')
#     label = np.load(labels)
#     boundary = np.load(boundaries)
#     num, cnt = np.unique(label, return_counts=True)
#     b, ct = np.unique(boundary, return_counts=True)
#     for n, c in zip(num, cnt):
#         nums[n] += c
#     for i, j in zip(b, ct):
#         # print(i,j)
#         bounds[i] += j
#
# print(nums, bounds)
#
# class_num = torch.tensor(nums)
# total = class_num.sum().item()
# frequency = class_num.float() / total
# median = torch.median(frequency)
# class_weight = median / frequency
# print(class_weight)
#
# pos_ratio = bounds[1] / sum(nums)
# pos_weight = 1 / pos_ratio
#
# pos_num = torch.tensor(bounds)
# totalb = pos_num.sum().item()
# frequencyb = pos_num.float() / totalb
# medianb = torch.median(frequencyb)
# pos_weight = medianb / frequencyb
# print(pos_weight)


# def main():
#     world_size = torch.cuda.device_count()
#     mp.spawn(train_ddp, args=(world_size, train_loader, model, criterion_cls, criterion_bound, lambda_bound_loss, optimizer, epoch, device), nprocs=world_size, join=True)




def train_ddp(rank, world_size, train_loader, model, criterion_cls, criterion_bound, lambda_bound_loss, optimizer, epoch, device):
    setup_ddp(rank, world_size)


    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


    # Your training logic goes here


    cleanup_ddp()




import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):


        score = -val_loss


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
