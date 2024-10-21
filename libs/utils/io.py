import cv2
import numpy as np
import pickle






def get_video_prop(path):
    """Get properties of a video"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return dict(fps=fps, num_frames=num_frames, height=height, width=width)






def load_numpy(fpath):
    return np.load(fpath)


def write_numpy(fpath, value):
    return np.save(fpath, value)


def load_pickle(fpath):
    return pickle.load(open(fpath, 'rb'))


def write_pickle(fpath, value):
    return pickle.dump(value, open(fpath, 'wb'))

def load_labels(ground_truth_file,prediction_file):
    with open(ground_truth_file, 'r') as gt_file:
        groun_truth= [int(line.strip()) for line in gt_file]

    with open(prediction_file,'r') as pred_file:
        predictions= [int(line.strip()) for line in pred_file]

    return groun_truth,predictions

def generate_colors(num_classes):
    np.random.seed(42)
    return np.random.randint(0,255,size=(num_classes,3))

def draw_progress_bars(frame,ground_truth,prediction,progress_gt,progress_pred,label_dict,colors):
     height, width, _ = frame.shape

    # Draw Ground Truth Progress Bar
     gt_bar_width = int(progress_gt * width)
     cv2.rectangle(frame, (0, height-60), (gt_bar_width, height-30), colors[ground_truth], -1)
     cv2.putText(frame, f'GT: {label_dict[ground_truth]}', (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw Prediction Progress Bar
     pred_bar_width = int(progress_pred * width)
     cv2.rectangle(frame, (0, height-30), (pred_bar_width, height), colors[prediction], -1)
     cv2.putText(frame, f'Pred: {label_dict[prediction]}', (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

     return frame

def create_progress_video(input_video, ground_truth,predictions,label_dict,output_file,fps=30,resolution=(640,480)):
    num_classes = len(label_dict)  # Number of unique classes
    colors = generate_colors(num_classes)

    video_cap = cv2.VideoCapture(input_video)

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames= int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


    # num_classes= len(set(ground_truth+ predictions))
    # colors= generate_colors(num_classes)

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file,fourcc,fps,(width,height))


    num_frames_gt = len(ground_truth)
    num_frames_pred = len(predictions)

    frame_idx=0 

    while video_cap.isOpened():
        ret,frame= video_cap.read()
        

    # Generate each frame
    for frame_idx in range(max(num_frames_gt, num_frames_pred)):
        # Create a blank frame
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        # Calculate progress for ground truth and predictions
        progress_gt = min(frame_idx / num_frames_gt, 1.0)
        progress_pred = min(frame_idx / num_frames_pred, 1.0)

        # Get the ground truth and predicted labels for the current frame
        gt_label = ground_truth[min(frame_idx, num_frames_gt-1)]
        pred_label = predictions[min(frame_idx, num_frames_pred-1)]

        # Draw progress bars on the frame
        frame = draw_progress_bars(frame, gt_label, pred_label, progress_gt, progress_pred, label_dict, colors)

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

