import numpy as np
import cv2, random, sys, math, cProfile

def beauty_filter(img, colour_sigma=30, space_sigma=20, mode="warm"):

    image_y, image_x, image_channels = img.shape # Get dimensions of image to work with

    new_img = cv2.bilateralFilter(img, -1, colour_sigma, space_sigma) # Apply bilaterial filtering with given parameters

    red_adjust = {}
    blue_adjust = {} # Create lookup dictionaries for colour values


    if mode == "warm": # Generate colour values to "warm" image i.e. drop blue and increase red
        red_pow = 1.04
        blue_pow = 0.96 # Some powers for altering values
        for i in range(256): # For every possible red value
            red_adjust[i] = min(math.pow(i, red_pow), 255) # Raise to power and cap at 255
            blue_adjust[i] = max(math.pow(i, blue_pow), 0) # Raise to power and cap at 0
    elif mode == "cold": # Same as warm, but drop red and increase blue
        red_pow = 0.96
        blue_pow = 1.04
        for i in range(256):
            red_adjust[i] = max(math.pow(i, red_pow), 0)
            blue_adjust[i] = min(math.pow(i, blue_pow), 255)
    else: # If a wrong mode is specified, exit
        print("Invalid mode specified, aborting!")
        return

    for y in range(image_y):
        for x in range(image_x): # Iterate over the entire image
            new_img.itemset((y, x, 0), blue_adjust[new_img.item(y, x, 0)])
            new_img.itemset((y, x, 2), red_adjust[new_img.item(y, x, 2)]) # Replace the pixel value from the lookup table


    cv2.imshow("Beautification Filter", new_img) # Show image
    key = cv2.waitKey(0)

    if (key == ord('x')):
        cv2.destroyAllWindows()

args = sys.argv

if len(args) < 2:
    print("Specify image file.")
elif len(args) == 2:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    beauty_filter(img)
elif len(args) == 3:
    print("Give both sigma values or neither")
elif len(args) == 4:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    beauty_filter(img, int(args[2]), int(args[3]))
elif len(args) == 5:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    beauty_filter(img, int(args[2]), int(args[3]), args[4])