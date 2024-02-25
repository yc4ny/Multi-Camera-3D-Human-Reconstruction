# Copyright (c) 2024 Yonwoo Choi
# All rights reserved.
# This code is part of my research at Seoul National University.
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without direct permission from the author.

import os
import cv2
import numpy as np
import argparse

def parse_args():
    """
    Parse command line arguments for input directories and output directory.
    """
    parser = argparse.ArgumentParser(description="Concatenate images with overlays and error metrics.")
    parser.add_argument('--dlt_dir', type=str, required=True, help='Directory path for DLT images.')
    parser.add_argument('--nl_dir', type=str, required=True, help='Directory path for Non-Linear images.')
    parser.add_argument('--kp_dir', type=str, required=True, help='Directory path for 3D keypoints visualization images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for concatenated images.')
    parser.add_argument('--error_file', type=str, required=True, help='Numpy file containing error metrics.')
    return parser.parse_args()

def calculate_average_error(error_array):
    """
    Calculate the average error from a numpy array of errors.
    """
    return np.mean(error_array)

def add_text_overlay(image, text, position, font_scale, font_color, thickness=10, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Add text overlay to an image at a specified position.
    """
    cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

def process_images(args):
    """
    Main processing function to load, concatenate, and save images with overlays.
    """
    # Load error metrics
    dlt_error = np.load(os.path.join(args.error_file, 'dlt_error.npy'))
    nl_error = np.load(os.path.join(args.error_file, 'nl_error.npy'))
    dlt_error_average = calculate_average_error(dlt_error)
    nl_error_average = calculate_average_error(nl_error)

    # Get the number of images to process
    count = len([name for name in os.listdir(args.dlt_dir) if os.path.isfile(os.path.join(args.dlt_dir, name))])

    for i in range(1, count + 1):
        print(f"Concatenating image {i}...")

        # Load images
        dlt = cv2.imread(os.path.join(args.dlt_dir, f'{i}_reproject.jpg'))
        nl = cv2.imread(os.path.join(args.nl_dir, f'{i}_reproject.jpg'))
        kp_images = [cv2.imread(os.path.join(args.kp_dir, str(j), f'{i-1}_output.jpg')) for j in range(1, 5)]

        # Resize keypoint images
        kp_images_resized = [cv2.resize(kp, (3840, 2160), interpolation=cv2.INTER_CUBIC) for kp in kp_images]

        # Add text overlays
        add_text_overlay(dlt, 'Camera 1: DLT', (10, 150), 5, (0, 0, 255))
        add_text_overlay(nl, 'Camera 1: Non Linear Optimization', (10, 150), 5, (255, 0, 0))

        # Concatenate images horizontally and vertically
        im_h = cv2.hconcat([dlt, nl])
        kp_concat = cv2.vconcat([cv2.hconcat(kp_images_resized[:2]), cv2.hconcat(kp_images_resized[2:])])
        final = cv2.vconcat([im_h, kp_concat])

        # Save the final concatenated image
        cv2.imwrite(os.path.join(args.output_dir, f'{i}_concat.jpg'), final)

def main():
    args = parse_args()
    process_images(args)

if __name__ == "__main__":
    main()
