import os
import cv2
import glob
import numpy as np

dataset_names = ['BUSI1/train']
for dataset_name in dataset_names:

    root = '/opt/data/private/datasets/Medical/Ultrasound'

    edges_path = os.path.join(root, dataset_name, 'edges')
    masks_path = os.path.join(root, dataset_name, 'masks')

    if not os.path.exists(edges_path):
        os.makedirs(edges_path)

    for img_path in glob.glob(masks_path + '/*.png'):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, img_b = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
        img_ = ~img
        _, img_b2 = cv2.threshold(img_, 128, 2, cv2.THRESH_BINARY)
        contour, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(img)
        boundary = cv2.drawContours(boundary, contour, -1, 1, 16)
        
        img_b3 = img_b2 + boundary
        
        img_b3[img_b3==2] = 0
        img_b3[img_b3==3] = 0
        
        _, filename = os.path.split(img_path)
        
        cv2.imwrite(os.path.join(edges_path, filename), img_b3*255)
