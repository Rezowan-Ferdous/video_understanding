import os,torch
from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from typing import Any, Dict, List, Optional
import torch.nn.functional as F


from libs.utils.preprocess import ReducedKernelConv,extract_features_dinov2,extract_features




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_reducer = ReducedKernelConv(device=device)
feature_reducer = nn.DataParallel(feature_reducer)
# dataframe = create_dataframes(base_test_dir,video_filename,feature_filename,annot_filename)


pretrained_model = extract_features_dinov2




import albumentations as A
from albumentations.pytorch import ToTensorV2
transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# jigsaw_root= "/home/ubuntu/Dropbox/Datasets/jigsaw/jigsaw_dataset"
#
# jigsaw_path = "/home/local/data/rezowan/datasets/jigsaw"
actiondict = {'G1': 0,  # Reaching for needle with right hand
              'G2': 1,  # Positioning needle
              'G3': 2,  # Pushing needle through tissue
              'G4': 3,  # Transferring needle from left to right
              'G5': 4,  # Moving to center with needle in grip
              'G6': 5,  # Pulling suture with left hand
              'G7': 6,  # Pulling suture with right hand
              'G8': 7,  # Orienting needle
              'G9': 8,  # Using right hand to help tighten suture
              'G10': 9,  # Loosening more suture
              'G11': 10,  # Dropping suture at end and moving to end points
              'G12': 11,  # Reaching for needle with left hand
              'G13': 12,  # Making C loop around right hand
              'G14': 13,  # Reaching for suture with right hand
              'G15': 14, }  # Pulling suture with both hands




def extract_video_identifier(video_file):
    return '_'.join(video_file.split('/')[-1].split('_')[:-1])
    # Split the filename and remove the capture part (e.g., Needle_Passing_B001 from Needle_Passing_B001_capture1)




# Function to extract the core identifier from the annotation file
def extract_annotation_identifier(annotation_file):
    # Get the annotation identifier (e.g., Needle_Passing_B001 from Needle_Passing_B001.txt)
    return annotation_file.split('/')[-1].replace('.txt', '')




def jigsaw_files_df(jigsaw_root):
    annot_files = []
    video_files = []
    # feature_files=[]
    # boundary_files=[]
    # label_files=[]
    for folders in os.listdir(jigsaw_root):
        pathname = os.path.join(jigsaw_root, folders)
        if os.path.isdir(pathname):
            for folder in os.listdir(pathname):
                folder_path = os.path.join(pathname, folder)


                if folder == 'transcriptions':
                    annotations = os.listdir(folder_path)
                    annotations.sort()
                    # Store absolute paths for annotation files
                    for annot in annotations:
                        annot_files.append(os.path.join(folder_path, annot))


                if folder == 'video':
                    videos = os.listdir(folder_path)
                    videos.sort()
                    # Store absolute paths for video files
                    for video in videos:
                        video_files.append(os.path.join(folder_path, video))
                # if folder =='features':
                #     features = os.listdir(folder_path)
                #     features.sort()
                #     # Store absolute paths for video files
                #     for feature in features:
                #         feature_files.append(os.path.join(folder_path, feature))
                #


    # Create a mapping of annotation identifier to annotation file
    annotation_dict = {extract_annotation_identifier(annotation): annotation for annotation in annot_files}


    # For each video, find the corresponding annotation by the identifier without the capture part
    matched_annotations = []


    for video in video_files:
        video_identifier = extract_video_identifier(video)
        annotation = annotation_dict.get(video_identifier, None)
        matched_annotations.append(annotation)


    # Check the lengths of video_files and matched_annotations
    print(f"Length of video_files: {len(video_files)}")
    print(f"Length of matched_annotations: {len(matched_annotations)}")


    # Ensure both lists have the same length
    if len(video_files) == len(matched_annotations):
        # Create a DataFrame
        jigsawdf = pd.DataFrame({
            'video_file': video_files,  # Ensure the correct video list is used
            'annotation_path': matched_annotations
        })
    else:
        print("Error: The lengths of video_files and matched_annotations do not match.")
    return jigsawdf




def calculate_jigsaw_segments(annotation_path, actiondict, sample_rate):
    segments = []
    with open(annotation_path, "r") as f:
        contents = f.read().split("\n")[:-1]
    endf = int(contents[-1].strip().split()[1])


    cal_len = 0
    labels = []
    boundaries = []
    total_len = 0
    if contents[0].split()[0] != 0:
        first = int(contents[0].split()[0])
        indc = np.arange(0, first - 1, sample_rate)
        efl = len(indc)
        unlabeled = np.full(efl, -100)
        labels.extend(unlabeled)
        boundaries.extend(unlabeled)


    for content in contents:
        st, ed, act = content.split()
        start_frame = int(st)
        end_frame = int(ed) - 1  # End frame inclusive
        segment_length = end_frame - start_frame + 1  # Include the last frame


        label = actiondict[act]
        total_len += segment_length


        # Calculate effective indices ensuring the last frame is included
        # effective_indices = np.linspace(0, segement_length-1, num=(segement_length + sample_rate - 1) // sample_rate, dtype=int)
        effective_indices = np.arange(0, segment_length - 1, sample_rate)
        effective_length = len(effective_indices)


        cal_len += effective_length


        # Fill labels and boundaries
        seg = np.full(effective_length, label)
        bound = np.full(effective_length, 0)  # Default intermediate


        if effective_length > 10:
            bound[0:5] = 1  # Start boundary
            bound[-5:] = 2  # End boundary
        elif effective_length <= 10:
            mid_point = effective_length // 2
            bound[0:mid_point] = 1  # Start boundary for first half
            bound[mid_point:] = 2  # End boundary for second half


        # bound[0:5]= 1;bound[5: effective_length-5]=0; bound[effective_length-5:effective_length+1]=2;
        labels.extend(seg)
        boundaries.extend(bound)
        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'phase': act,
            'label': actiondict[act]
        })
    labels_arr = np.array(labels)
    bounds_arr = np.array(boundaries)
    return segments, labels_arr, bounds_arr, endf




def make_full_df(files_df, actiondict, sample_rate):
    num_frames = []
    allsegmetns = []
    alllabels = []
    allboundaries = []
    feature_path = []
    video_ids=[]




    for i, r in files_df.iterrows():
        annot_file = r['annotation_path']
        video_file = r['video_file']
        vid_name = video_file.split('/')[-1].split('.')[0]


        class_root_path = Path(video_file).parents[1]
        f_path = str(class_root_path) + '/features'
        l_path = str(class_root_path) + '/labels'
        b_path = str(class_root_path) + '/boundaries'
        feat_path = f_path + '/' + vid_name + '.npy'
        label_path= l_path + '/'+str(sample_rate)+'_' + vid_name + '.npy'
        bound_path= b_path + '/'+str(sample_rate)+'_' + vid_name + '.npy'
        feature_path.append(feat_path)
        video_ids.append(vid_name)


        # vid= get_video_prop(video_file)
        # total_frames= vid['num_frames']
        jigsaw_segments, label, bound, num_frame = calculate_jigsaw_segments(annot_file, actiondict,
                                                                             sample_rate=sample_rate)
        np.save(label_path,label)
        np.save(bound_path,bound)


        num_frames.append(num_frame)
        allsegmetns.append(jigsaw_segments)
        alllabels.append(label_path)
        allboundaries.append(bound_path)


    files_df['feature_path'] = feature_path
    files_df['frames'] = num_frames
    files_df['segments'] = allsegmetns
    files_df['labels'] = alllabels
    files_df['boundaries'] = allboundaries
    files_df['video_id'] =video_ids
    full_df = files_df
    return full_df






# sample_rate = 1
# jigsaw_files_df = jigsaw_files_df(jigsaw_path)
# jigsaw_full_df = make_full_df(jigsaw_files_df, actiondict, sample_rate)
# extract_features(df=jigsaw_full_df,pretrained_model=pretrained_model,feature_reducer=feature_reducer,batch_size=320)


class JigsawDataset(Dataset):


    def __init__(self, dataframe, root, num_classes, action_dict, sample_rate=1):
        self.df = dataframe
        self.root = root
        self.num_class = num_classes
        self.action_dict = action_dict
        self.sample_rate = sample_rate


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        feat_file, annot_file, labels, boundaries, segments, frames, vid = (
            row['feature_path'], row['annotation_path'], row['labels'], row['boundaries'], row['segments'],
            row['frames'], row['video_id'])
        features = np.load(feat_file).astype(np.float32)
        max_index = features.shape[0]
        labels = np.load(labels)
        boundaries = np.load(boundaries)
        indices = np.linspace(0, max_index - 1, labels.shape[0]).astype(int)
        #
        # # Sample the sequence, label, and mask using the generated indices
        features = features[indices, :]  # Select along the time dimension


        # Ensure that all arrays have the same length initially
        assert features.shape[0] == labels.shape[0] == boundaries.shape[0], "Initial lengths are not the same!"


        # # Sample the features, labels, and boundaries with the same rate
        # sampled_features = feature[::self.sample_rate, :]  # Sample along the first dimension
        # sampled_labels = labels[::self.sample_rate]  # Sample the labels
        # sampled_boundaries = boundaries[::self.sample_rate]  # Sample the boundaries


        sample = {
            "feature": features,
            "label": labels,
            "feature_path": feat_file,
            "boundary": boundaries,
        }
        return sample


    def resampled_data(self, num_frames, fps, annotation_path, feature_file, sample_rate):
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


        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)
        boundary = torch.from_numpy(boundary)
        # print("shape length",t)
        if pad_t > 0:
            feature = F.pad(feature, (0, pad_t), mode="constant", value=0.0)
            label = F.pad(label, (0, pad_t), mode="constant", value=255)
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








