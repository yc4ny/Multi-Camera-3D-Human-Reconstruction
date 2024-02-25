# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import os
import cv2
import numpy as np
import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='Concatenate images from multiple camera angles with overlays.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory path for input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory path for output concatenated images.')
    return parser.parse_args()

def ensure_output_dir_exists(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def process_images(input_dir, output_dir, count, on_states):
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (10, 150)
    topRightCornerOfText = (3200, 150)
    fontScale1 = 5
    fontColor1 = (0, 0, 255)  # Red for camera labels
    fontColor3 = (0, 255, 0)  # Green for 'ON' label
    thickness = 10
    lineType = 2

    for i in range(1, count + 1):
        print(f"Processing image {i}...")
        images = []
        for c in range(1, 6):
            img_path = os.path.join(input_dir, f'nonlinear_C{c}/{i}.jpg')
            img = cv2.imread(img_path)
            cv2.putText(img, f'Camera {c}', topLeftCornerOfText, font, fontScale1, fontColor1, thickness, lineType)
            if on_states[c-1][i-1] == 1:
                cv2.putText(img, 'ON', topRightCornerOfText, font, fontScale1, fontColor3, thickness, lineType)
            images.append(img)

        # Concatenate images
        c1_c2 = cv2.hconcat([images[0], images[1]])
        c3_c4 = cv2.hconcat([images[2], images[3]])
        c1_c2_c3_c4 = cv2.vconcat([c1_c2, c3_c4])
        images[4] = cv2.resize(images[4], dsize=(c1_c2_c3_c4.shape[1], images[4].shape[0]), interpolation=cv2.INTER_CUBIC)
        final_image = cv2.hconcat([c1_c2_c3_c4, images[4]])

        # Save the concatenated image
        cv2.imwrite(os.path.join(output_dir, f'{i}_concat.jpg'), final_image)

def main():
    args = setup_argparse()
    ensure_output_dir_exists(args.output_dir)
    on_states = [np.load(os.path.join(args.input_dir, f'checkerboard_on/C{i}_initial.npy')) for i in range(1, 6)]
    count = len([name for name in os.listdir(os.path.join(args.input_dir, 'nonlinear_C1')) if os.path.isfile(os.path.join(args.input_dir, 'nonlinear_C1', name))])
    process_images(args.input_dir, args.output_dir, count, on_states)

if __name__ == "__main__":
    main()
