import numpy as np
import pandas as pd


from libs.utils.io import get_video_prop
from typing import Any, Dict, List, Optional


from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torch,os
import torch.nn.functional as F


cholec_root= "/home/ubuntu/Dropbox/Datasets/cholec80"


PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging"
]


action_dict= {"Preparation":0,
    "CalotTriangleDissection":1,
    "ClippingCutting":2,
    "GallbladderDissection":3,
    "GallbladderRetraction":4,
    "CleaningCoagulation":5,
    "GallbladderPackaging":6}


INSTRUMENTS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]


def get_base_name(file):
    return file.split('-timestamp')[0].split('.')[0]




# cholec_root = "/home/local/data/rezowan/datasets/cholec/"


action_dict = {"Preparation": 0,
               "CalotTriangleDissection": 1,
               "ClippingCutting": 2,
               "GallbladderDissection": 3,
               "GallbladderRetraction": 4,
               "CleaningCoagulation": 5,
               "GallbladderPackaging": 6}




def calculate_segments(phase_file, action_dict, sample_rate=1):
    segments = []  # List to hold the segments
    start_frame = 0  # Starting frame for the first segment
    prev_phase = None  # Previous phase to track changes


    # if dataset=="cholec":
    with open(phase_file, "r") as f:
        gt = f.read().split("\n")[1:-1]
    # Total number of frames
    num_frames = len(gt)


    # Create effective frame indices based on the sample rate
    effective_indices = np.arange(0, num_frames, sample_rate)
    effective_length = len(effective_indices)


    # Create arrays to store ground truth phase information and boundaries
    gt_array = np.full(effective_length, -100)
    boundary_array = np.zeros(effective_length)


    for i in range(num_frames):
        frame_data = gt[i].split("\t")
        current_frame = int(frame_data[0])  # Frame number
        current_phase = frame_data[-1]  # Phase name
        if current_frame in effective_indices:
            effective_index = np.where(effective_indices == current_frame)[0][0]


            # Process only if the current phase is in the action dictionary
            if current_phase in action_dict:
                gt_array[effective_index] = action_dict[current_phase]


                # Track phase changes
                if prev_phase is None:
                    # Initialize the previous phase
                    prev_phase = current_phase
                    start_frame = effective_index  # Start segment at the first frame


                elif current_phase != prev_phase:
                    # When phase changes, record the previous phase segment
                    end_frame = effective_index - 1


                    # When phase changes, record the previous phase segment
                    segments.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'phase': prev_phase,
                        'label': action_dict[prev_phase]
                    })


                    # Define boundary conditions: First 5 frames as start, last 5 frames as end
                    boundary_array[start_frame:start_frame + 5] = 1  # Start boundary
                    boundary_array[end_frame - 4:end_frame + 1] = 2  # End boundary


                    # All intermediate frames between start and end are refined (mark as 0)
                    boundary_array[start_frame + 5:end_frame - 4] = 0


                    start_frame = effective_index
                    prev_phase = current_phase


    # After the loop, handle the last active segment
    if prev_phase is not None:
        end_frame = effective_index


        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'phase': prev_phase,
            'label': action_dict[prev_phase]
        })


        # Define boundary for the last segment
        boundary_array[start_frame:start_frame + 5] = 1  # Start boundary
        boundary_array[end_frame - 4:end_frame + 1] = 2  # End boundary
        if end_frame - 4 >= start_frame + 5:
            boundary_array[start_frame + 5:end_frame - 4] = 0






    return segments, gt_array, boundary_array




def create_cholec_df(cholec_root, action_dict, sample_rate=1):
    phases_path = os.path.join(cholec_root, 'phase_annotations')
    tools_path = os.path.join(cholec_root, 'tool_annotations')
    features_path = os.path.join(cholec_root, 'features')
    videos_path = os.path.join(cholec_root, 'videos')
    labels_path= os.path.join(cholec_root,'labels')
    boundaries_path= os.path.join(cholec_root,'boundaries')


    # for folds in folders:
    phase_list = os.listdir(phases_path)
    tool_list = os.listdir(tools_path)
    feature_list = os.listdir(features_path)
    video_list = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    phase_list.sort()
    tool_list.sort()
    feature_list.sort()
    video_list.sort()
    feat_files = []
    phase_files = []
    tool_files = []
    segments = []
    labels = []
    boundaries = []
    vid_id=[]


    video_files = []
    for feat in zip(feature_list, tool_list, phase_list, video_list):
        base_name = feat[0].split('.')[0]
        label_path = os.path.join(labels_path,str(sample_rate)+'_'+base_name+'.npy')
        bound_path= os.path.join(boundaries_path,str(sample_rate)+'_'+base_name+'.npy')


        if feat[1].startswith(base_name) and feat[2].startswith(base_name):
            phase_file = os.path.join(phases_path, feat[2])
            tool_file = os.path.join(tools_path, feat[1])
            feat_file = os.path.join(features_path, feat[0])
            vid_file = os.path.join(videos_path, feat[3])
            segment, gt, bound = calculate_segments(phase_file, action_dict, sample_rate=sample_rate)
            segments.append(segment)


            if os.path.isfile(feat_file):
                feat_files.append(feat_file)
            if os.path.isfile(phase_file):
                phase_files.append(phase_file)
            if os.path.isfile(tool_file):
                tool_files.append(tool_file)
            if os.path.isfile(vid_file):
                video_files.append(vid_file)




            np.save(label_path, gt)
            np.save(bound_path, bound)
            labels.append(label_path)
            boundaries.append(bound_path)
            vid_id.append(base_name)


    # Create a dataframe with the relevant paths
    df = pd.DataFrame({
        'video_id': vid_id,
        'video_file': video_files,
        'annotation_path': phase_files,
        'tool_annotation': tool_files,
        'feature_path': feat_files,
        'segments': segments,
        'labels': labels,
        'boundaries': boundaries,
        'fps':25,
        'frames':len(labels),


    })
    return df




# cholec_df = create_cholec_df(cholec_root, action_dict, sample_rate=1)


class CholecDataset(Dataset):
    def __init__(self,dataframe,root,num_classes,action_dict,sample_rate=1):
        self.df=dataframe
        self.root=root
        self.num_class=num_classes
        self.action_dict= action_dict
        self.sample_rate=sample_rate


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        feat_file,annot_file,labels,boundaries,segments,frames,vid,fps =(
            row['feature_path'],row['annotation_path'],row['labels'],row['boundaries'],row['segments'],row['frames'],row['video_id'],row['fps'])
        feature = np.load(feat_file).astype(np.float32)
        max_index= feature.shape[0]
        labels= np.load(labels)
        boundaries= np.load(boundaries)
        indices = np.linspace(0, max_index - 1, labels.shape[0]).astype(int)
        #
        # # Sample the sequence, label, and mask using the generated indices
        feature = feature[ indices,:]  # Select along the time dimension




        # Ensure that all arrays have the same length initially
        assert feature.shape[0] == labels.shape[0] == boundaries.shape[0], "Initial lengths are not the same!"




        # Sample the features, labels, and boundaries with the same rate
        # if self.transform is not None:
        #     feature, label, boundary = self.transform([feature, label, boundary])


        sample = {
            "feature": feature,
            "label": labels,
            "feature_path": feat_file,
            "boundary": boundaries,
            "segments": segments,
        }
        return sample


    def segments_features(self, segments, features,labels):
        last_frame=segments[-1][1]
        labels_len= labels.shape[0]
        featue_len= features.shape[0]
        # for segment in segments:
        feat_indeces= np.linspace(0,featue_len-1,labels_len).astype(int)
        features= features[feat_indeces,:]
        return  features






    def resampled_data(self,num_frames, fps, annotation_path, feature_file, sample_rate):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()


        segments = []
        total_frames = num_frames
        lastframe = 0


        # Initialize arrays for tracking sampled frames across segments
        reduced_labels = []
        reduced_boundaries = []


        for line in lines:
            # Split line by ',' and extract the start_frame, end_frame, and action
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # Skip malformed lines


            start_frame, end_frame, action = parts
            try:
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                action = int(action)  # Assuming action is an integer


                # Ensure the end_frame does not exceed total number of frames
                end_frame = min(end_frame, total_frames - 1)


                # Apply sampling rate within the segment
                sampled_frames = np.arange(start_frame, end_frame + 1, sample_rate)


                # Append the action to reduced_labels for sampled frames
                reduced_labels.extend([action] * len(sampled_frames))


                # Handle boundary assignment for sampled frames
                sampled_length = len(sampled_frames)


                if sampled_length >= 10:  # If the segment is long enough
                    # Start boundary: 1, Middle: 0, End boundary: 2
                    reduced_boundaries.extend([1] * min(5, sampled_length))  # Start boundary
                    reduced_boundaries.extend([0] * (sampled_length - 10))  # Inside action
                    reduced_boundaries.extend([2] * min(5, sampled_length))  # End boundary
                else:
                    # Short segment: Mark entire segment with boundary
                    reduced_boundaries.extend([1] * (sampled_length - 1))  # Start and mid boundary
                    reduced_boundaries.append(2)  # End boundary


            except ValueError:
                continue  # Skip lines with invalid data


            frame_length = end_frame - start_frame
            duration = frame_length / fps


            # Store segment information
            segments.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'label': action,
                'duration': duration
            })


        # After processing all segments, convert lists to numpy arrays
        labels = np.array(reduced_labels)
        boundaries = np.array(reduced_boundaries)




def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([s["feature"].shape[0] for s in sample])


    feat_list = []
    label_list = []
    path_list = []
    boundary_list = []
    length_list = []


    for s in sample:
        feature = s["feature"]
        label = s["label"]
        boundary = s["boundary"]
        feature_path = s["feature_path"]


        feature = feature.T
        _, t = feature.shape
        pad_t = max_length - t


        length_list.append(t)


        feature= torch.from_numpy(feature)
        label= torch.from_numpy(label)
        boundary=torch.from_numpy(boundary)
        # print("shape length",t)
        if pad_t > 0:
            feature = F.pad(feature, (0, pad_t), mode="constant", value=0.0)
            label = F.pad(label, (0, pad_t), mode="constant", value=-100)
            boundary = F.pad(boundary, (0, pad_t), mode="constant", value=0.0)


        # reshape boundary (T) => (1, T)  / boundary.unsqueeze(0)
        # boundary = torch.from_numpy(boundary).unsqueeze(0)


        # label= torch.from_numpy(label).unsqueeze(0)
        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)
        # boundary_list.append(boundary)
        boundary_list.append(boundary)


    # print(feat_list)
    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)


    # merge labels from tuple of 2D tensor to 3D tensor
    # shape (N, 1, T)
    boundaries = torch.stack(boundary_list, dim=0)


    # generate masks which shows valid length for each video (N, 1, T)
    masks = [[1 if i < length else 0 for i in range(max_length)] for length in length_list]
    masks = torch.tensor(masks, dtype=torch.bool)
    # print("mask , feature labesl and boundary shape for training ",masks.shape, features.shape,labels.shape,boundaries.shape)


    return {
        "feature": features,
        "label": labels,
        "boundary": boundaries,
        "feature_path": path_list,
        "mask": masks,
    }


