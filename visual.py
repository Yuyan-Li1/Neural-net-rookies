import cv2
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(30, 30))
rows = 2
columns = 2
# load same image and label from first training set 02
sample_label_1 = cv2.imread('panoptic_maps/train/0002/000480.png')
sample_image_1 = cv2.imread('step_images/train/STEP-ICCV21-02/000480.jpg')
# load same image and label from second training set 09
sample_label_2 = cv2.imread('panoptic_maps/train/0009/000250.png')
sample_image_2 = cv2.imread('step_images/train/STEP-ICCV21-09/000250.jpg')
fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(sample_image_1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Original")
fig.add_subplot(rows, columns, 2)
plt.imshow(cv2.cvtColor(sample_label_1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Ground Truth")
fig.add_subplot(rows, columns, 3)
plt.imshow(cv2.cvtColor(sample_image_2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Original")
fig.add_subplot(rows, columns, 4)
plt.imshow(cv2.cvtColor(sample_label_2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Ground Truth")
# plt.show()
Blue_channel = sample_label_1[:, :, 0]
Red_channel = sample_label_1[:, :, 2]
print('Red_channel:')
print(np.unique(Red_channel))
print('Blue_channel:')
print(np.unique(Blue_channel))
# next we explore if how the red and blue channel looks like visually
fig = plt.figure(figsize=(30, 30))
rows = 1
columns = 3
fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(sample_image_1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Original")
fig.add_subplot(rows, columns, 2)
plt.imshow(Red_channel, cmap='gray')
plt.axis('off')
plt.title("Red/ Semantic Ids/ Classes")
fig.add_subplot(rows, columns, 3)
plt.imshow(Blue_channel, cmap='gray')
plt.axis('off')
plt.title("Blue / Instance ID / People")
# plt.show() this part is to confirm that the instance IDs are only from the class 4 (person) , if the function does
# not print anything then all instances are for persons only
for i in range(Blue_channel.shape[0]):
    for j in range(Blue_channel.shape[0]):
        if Red_channel[i][j] != 4 and Blue_channel[i][j] != 0:
            print("object id is")
            print(Blue_channel[i][j])
            print("class id is")
            print(Red_channel[i][j])

# we use open cv boudning rectangle and rectangle functions to draw the the bounding boxes
instance_IDs = np.unique(Blue_channel)
# we count the number of instance ids in the image and this reflects the number of people, we must subtract 1 because
# instance id 0 is for the background (black part)
print("the count of people in this image is {}".format(len(instance_IDs) - 1))
for tracking_ID in instance_IDs:
    if tracking_ID != 0:
        c = Blue_channel.copy()
        c = np.where(c == tracking_ID, 200, 0).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(sample_image_1, (x, y), (x + w, y + h), (0, 0, 255), 3)
plt.rcParams['figure.figsize'] = [30, 30]
plt.imshow(cv2.cvtColor(sample_image_1, cv2.COLOR_BGR2RGB))
plt.show()
# we do the same for the other training folder 09
Blue_channel = sample_label_2[:, :, 0]
Red_channel = sample_label_2[:, :, 2]
# we use open cv boudning rectangle and rectangle functions to draw the the bounding boxes
instance_IDs = np.unique(Blue_channel)
# we count the number of instance ids in the image and this reflects the number of people, we must subtract 1 because
# instance id 0 is for the background (black part)
print("the count of people in this image is {}".format(len(instance_IDs) - 1))
for tracking_ID in instance_IDs:
    if tracking_ID != 0:
        c = Blue_channel.copy()
        c = np.where(c == tracking_ID, 200, 0).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(sample_image_2, (x, y), (x + w, y + h), (0, 0, 255), 3)
plt.rcParams['figure.figsize'] = [30, 30]
plt.imshow(cv2.cvtColor(sample_image_2, cv2.COLOR_BGR2RGB))
plt.show()
