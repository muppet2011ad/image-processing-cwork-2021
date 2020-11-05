import numpy as np
import cv2, random, sys, math, cProfile

img = cv2.imread("input_2.jpg", cv2.IMREAD_COLOR)
image_y, image_x, image_channels = img.shape

new_img = cv2.bilateralFilter(img, -1, 30, 20)

mode = "warm"

red_adjust = {}
blue_adjust  ={}

if mode == "warm":
    for i in range(256):
        red_adjust[i] = min(i*1.2, 255)
        blue_adjust[i] = max(i*0.9, 0)
elif mode == "cold":
    for i in range(256):
        red_adjust[i] = max(i*0.9, 0)
        blue_adjust[i] = min(i*1.1, 255)

for y in range(image_y):
    for x in range(image_x):
        new_img.itemset((y, x, 0), blue_adjust[new_img.item(y, x, 0)])
        new_img.itemset((y, x, 2), red_adjust[new_img.item(y, x, 2)])


cv2.imshow("Beautification Filter", new_img)
key = cv2.waitKey(0)

if (key == ord('x')):
    cv2.destroyAllWindows()