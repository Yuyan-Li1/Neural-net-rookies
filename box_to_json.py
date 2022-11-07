import json
import os

import cv2
import numpy as np


def main():
    label_path = 'panoptic_maps/train/0009'
    save_path = 'step_images/train/STEP-ICCV21-09'
    for label_filename in os.listdir(label_path):
        # print(label_filename)
        label = cv2.imread(f'{label_path}/{label_filename}')
        blue_channel = label[:, :, 0]
        output_filename = label_filename[:-4]
        # print(output_filename)
        save_to_json(save_path, output_filename, blue_channel)


def save_to_json(save_path, filename, blue_channel):
    rectangles = []
    instance_IDs = np.unique(blue_channel)
    for tracking_ID in instance_IDs:
        if tracking_ID != 0:
            c = blue_channel.copy()
            c = np.where(c == tracking_ID, 200, 0).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(c)
            rectangle = {
                'x': x,
                'y': y,
                'width': w,
                'height': h
            }
            rectangles.append(rectangle)
    with open(f'{save_path}/{filename}.json', 'w') as out:
        out.write(json.dumps(rectangles))
        print(f'Coordinates in {filename} saved to json')


if __name__ == '__main__':
    main()
