import cv2
import numpy as np

# Load the ground truth lane video and the output video
cap_ground_truth = cv2.VideoCapture('detected_lane.mp4')
cap_output = cv2.VideoCapture('output_video.mp4')

# Function to calculate Intersection over Union (IoU)
def calculate_iou(detected_lane, ground_truth_lane):
    intersection = np.logical_and(detected_lane, ground_truth_lane)
    union = np.logical_or(detected_lane, ground_truth_lane)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Variables to keep track of cumulative IoU and frame count
cumulative_iou = 0
frame_count = 0

while cap_ground_truth.isOpened() and cap_output.isOpened():
    ret_ground_truth, frame_ground_truth = cap_ground_truth.read()
    ret_output, frame_output = cap_output.read()

    if not ret_ground_truth or not ret_output:
        break

    # Convert frames to grayscale if necessary
    frame_ground_truth = cv2.cvtColor(frame_ground_truth, cv2.COLOR_BGR2GRAY)
    frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2GRAY)

    # Calculate IoU for the current frame
    iou = calculate_iou(frame_output, frame_ground_truth)

    # Add the IoU value to the cumulative IoU
    cumulative_iou += iou
    frame_count += 1

# Calculate the average IoU across all frames
average_iou = cumulative_iou / frame_count

# Print the average IoU as the accuracy measure
print(f'Average Intersection over Union (IoU): {average_iou:.2f}')

# Release video captures
cap_ground_truth.release()
cap_output.release()
