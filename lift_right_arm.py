import os
import cv2
import time
import posenet
import tensorflow as tf
import argparse
from statistics import mean
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
#parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int,default=640) # default=1280)
parser.add_argument('--cam_height', type=int,default=480) # default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
# parser.add_argument('--scale_factor', type=float, default=1)

parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def posen(out_save_path, filename):
    start = time.time()
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = 16
        full_path = os.path.join(out_save_path, filename)
        print('Read video path ', full_path)
        cap = cv2.VideoCapture(full_path)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('model input frames ', frames_num)
        frame_count = 0
        all_frame = list()
        while (frames_num > frame_count):
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image})
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=17,
                min_pose_score=0.15)
            keypoint_coords *= output_scale
            frame = dict()
            frame['keypoint_coords'] = keypoint_coords[0]
            all_frame.append(frame)
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords, min_pose_score=0.15,
                min_part_score=0.1)
            frame_count += 1
            cv2.imshow('posenet', display_image)
            # frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end = time.time()
        out_time = end - start
        print('out_time ==> ', out_time)
        cv2.destroyAllWindows()
        return all_frame


out_save_path = './lift_arm_video'
out_name = []
filename = 'file_3461294_2021-04-30_12_32_09_4.avi' # paralysis

all_frames = posen(out_save_path, filename)

elbow_list = []
for frame in all_frames:
    val = int(frame['keypoint_coords'][8][0])
    elbow_list.append(val)

# print('elbow list', elbow_list)
elbow_rest = mean(elbow_list[0:10])
# print('elbow rest', elbow_rest)

out_list = []
for frame in all_frames:
    val = int(frame['keypoint_coords'][10][0])
    out_list.append(val)

# print('wrist point ', out_list)
rest_postion = mean(out_list[0:10])
# print('wrist rest ', rest_postion)


movement_score = []
for val in out_list:
    if val < rest_postion:
        movement_score.append(val)
   
print(movement_score)
movement_id = False
if len(movement_score) > 10:
    move_small = min(movement_score)
    move_max = max(movement_score)
    print('movement difference ', move_max - move_small)
    if move_max - move_small > 10  and move_max - move_small < 40:
        movement_id = True
    else:
        movement_id = False

    print('movement_score ', len(movement_score))
    print('movement_id', movement_id)

# paralysis detection
all_scores = []
for val in out_list:
    if val > rest_postion - 20:
        score = 4
        all_scores.append(score)

# print('all score', len(all_scores))
# Remove rest postion points   
new_out = []
for val in out_list:
    if val < rest_postion - 20:
        new_out.append(val)

# print(new_out)
# find smallest movement points
min_number_list = []
for i in range(10):
    small_num = min(out_list)
    out_list.remove(small_num)
    min_number_list.append(small_num)
    # print('length',len(out_list))
    
movment_point = mean(min_number_list)
new_score = []
# Normal and drift to bed
for val in new_out:
    if val < movment_point + 20:
        score = 0
        new_score.append(score)
# print('cool')    
        
second_out = []
for val in new_out:
    if val > movment_point + 20:
        second_out.append(val)

# print(second_out)
another_move_score = []
for num in second_out:
    # print('num 1', num)
    i = 0
    # num = 170
    for index, check in enumerate(second_out):
        # print('check ', check)
        if num > check-20 and num < check+20:
            i = i + 1 
            # print('i ', i)
        if index == len(second_out)-1:
            # print('end')
            another_move_score.append(i)

print('another move score ', another_move_score)
# print('another move score ', len(another_move_score))

drifting = 0
for num in another_move_score:
    if num > 50:
        drifting = 1
        
count = Counter(all_scores)
print('count: ', count)
count2 = Counter(new_score)
print('count2', count2)

if count.get(4) and count[4] > len(all_frames) - 10 and movement_id == False:
    print('Complete Paralysis')
elif drifting == 1:
    print('Drift')
elif len(movement_score) > 5 and movement_id == True:
    print('Movement')
elif count2.get(0) and count2[0] > 170:
    print('Normal')
elif count2.get(0) and count2[0] > 20 and count2[0] < 170:
    print('Bed')
