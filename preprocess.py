import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def augmentation(train_src, label_src, train_dst, label_dst, augmentation_types):
    def normalize(img):
        # Set the norm matrix as 800 by 800 matrix
        norm = np.zeros((800,800))

        # return the normalized_img without modifying the original image
        return cv2.normalize(img.copy(),  norm, 0, 255, cv2.NORM_MINMAX)

    def blur(img, ksize=(10,10)):
        return cv2.blur(img.copy(), ksize)

    def flip(img, gt_img):
        output_img = cv2.flip(img.copy(), flipCode=1)
        output_label = cv2.flip(gt_img.copy(), flipCode=1)
        return (output_img, output_label)

    def randomBoundingCutout(img, gt_img):
        img = img.copy()
        gt_img = gt_img.copy()

        # This implimentation assume that the blue-channel is the channel of the object to detect
        blue_channel = gt_img[:,:,0]
        instance_IDs = np.unique(blue_channel)

        if len(instance_IDs) > 1:
            rand_idx = random.randint(1, len(instance_IDs)-1)
        else:
            rand_idx = 0

        rand_id = instance_IDs[rand_idx]

        c = blue_channel.copy()
        c = np.where(c == rand_id, 200, 0).astype(np.uint8)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h//2), (0,0,0), -1)
        cv2.rectangle(gt_img, (x, y), (x + w, y + h//2), (0,0,0), -1)

        return (img, gt_img)

    # get file in
    train_folders = os.listdir(train_src)
    label_folders = os.listdir(label_src)

    # Process each type of augementation one at a time
    for type in augmentation_types:
        # Process the train/label folders (folder-by-folder)
        for img_folder in train_folders:
            # If the corresponding label dir does not exist then skip
            if img_folder not in label_folders:
                print(f"Train folder {img_folder} not in label folders")
                continue

            # Get path of destinations
            train_dst_path = os.path.join(os.path.join(train_dst, f"{img_folder}_{type}"))
            label_dst_path = os.path.join(os.path.join(label_dst, f"{img_folder}_{type}"))
            if not os.path.exists(train_dst_path):
                os.mkdir(train_dst_path)

            # get all train files in the directory
            all_train_files = os.listdir(os.path.join(train_src, img_folder))
            train_imgs = list(filter(lambda file: (file.endswith(".jpg") or file.endswith(".png")), all_train_files))

            # get all label files in the directory
            all_label_files = os.listdir(os.path.join(label_src, img_folder))
            label_imgs = list(filter(lambda file: (file.endswith(".jpg") or file.endswith(".png")), all_label_files))

            if type not in ["flip", "cutout"]:
                shutil.copytree(os.path.join(label_src, img_folder), os.path.join(label_dst, f"{img_folder}_{type}"))
            elif not os.path.exists(label_dst_path):
                os.mkdir(label_dst_path)

            # Process image by image
            for train_img in train_imgs:
                print(f"Processing type {type} in {os.path.join(train_dst_path, train_img)}")
                # If train_img does not have a label then skip
                # (dodgy solution since train is in jpg and label is in png)
                if train_img.rstrip(".jpg")+".png" not in label_imgs:
                    print(f"Train image {train_img} not in label images")
                    print(label_imgs)
                    continue

                # Only alter label for cutout
                if type in ["flip", "cutout"]:
                    input_label = cv2.imread(os.path.join(label_src, img_folder, f'{train_img.rstrip(".jpg")}.png'))
                input_img = cv2.imread(os.path.join(train_src, img_folder, train_img))

                if type == "normalize":
                    output_img = normalize(input_img)
                elif type == "flip":
                    output_img, output_label = flip(input_img, input_label)
                    output_img = normalize(output_img)
                elif type == "blur":
                    output_img = blur(input_img)
                    output_img = normalize(output_img)
                elif type == "cutout":
                    output_img, output_label = randomBoundingCutout(input_img, input_label)
                    output_img = normalize(output_img)

                # Write output_image
                cv2.imwrite(os.path.join(train_dst_path, train_img), output_img)

                # Only write label for cut-out
                if type in ["flip", "cutout"]:
                    cv2.imwrite(os.path.join(label_dst_path, f"{train_img.rstrip('.jpg')}.png"), output_label)










def main():
    AUGMENTATION_TYPES = ["normalize", "flip", "blur", "cutout"]

    ROOT_DIR = "."
    path = os.path.join(ROOT_DIR)

    train_dir = os.path.join(ROOT_DIR, "train")
    augment_dir = os.path.join(ROOT_DIR, "augmentations")
    label_dir = os.path.join(ROOT_DIR, "labels", "train")


    # Check if train dir exists and not empty
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print(f"The directory train is either empty or does not exist")
        exit()

    # Check if label dir exists and not empty
    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        print(f"The directory label is either empty or does not exist")
        exit()

    # Check that labels exist for all dir in train
    train_folders = os.listdir(train_dir)
    label_folders = os.listdir(label_dir)
    if set(train_folders) != set(label_folders):
        print(f"Train folders {set(train_folders)} does not match label folders {set(label_folders)}")
        exit()

    # Check if augmentation dir exists
    if os.path.exists(augment_dir):
        choice = ""
        while choice.upper() not in ["Y", "N"]:
            choice = input("Augmentation Dir Already Exists: Do you want to delete it (Y/N)? ")
        if choice.upper() == "Y":
            shutil.rmtree(augment_dir)
        else:
            print("Exiting")
            exit()

    os.mkdir(augment_dir)

    # Copy the image from train/label to augmentations/train and augmentations/label
    if not os.path.exists(os.path.join(augment_dir, "train")):
        os.mkdir(os.path.join(augment_dir, "train"))
    if not os.path.exists(os.path.join(augment_dir, "label")):
        os.mkdir(os.path.join(augment_dir, "label"))

    train_dst = os.path.join(augment_dir, "train")
    label_dst = os.path.join(augment_dir, "label")



    augmentation(train_dir, label_dir, train_dst, label_dst, AUGMENTATION_TYPES)


if __name__ == "__main__":
    main()