import cv2
import os
import numpy as np
import imutils
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir  
imaPath = base_dir + '/left_sky'
output = base_dir + '/left_sky'
imaList = os.listdir(imaPath)
for files in imaList:
    path_ima = os.path.join(imaPath, files)
    path_processed = os.path.join(output, files)
    img = cv2.imread(path_ima, 0)

    mask = np.zeros_like(img)
    print(np.shape(img))


    ret, img = cv2.threshold(img, 127, 255 ,cv2.THRESH_BINARY)

    contours= cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area <= 5000:   #5000
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
            
    cv2.fillPoly(img, cv_contours, (0, 0, 0))
    cv2.imwrite(path_processed, img)


imaPath = base_dir + '/right_sky'
output = base_dir + '/right_sky'
imaList = os.listdir(imaPath)
for files in imaList:
    path_ima = os.path.join(imaPath, files)
    path_processed = os.path.join(output, files)
    img = cv2.imread(path_ima, 0)

    mask = np.zeros_like(img)
    print(np.shape(img))


    ret, img = cv2.threshold(img, 127, 255 ,cv2.THRESH_BINARY)

    contours= cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area <= 5000:   #5000
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
            
    cv2.fillPoly(img, cv_contours, (0, 0, 0))
    cv2.imwrite(path_processed, img)