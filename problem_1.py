import numpy as np
import cv2, random, sys, math, cProfile

windowName = "Light Leak Filter"

img = cv2.imread('./test.jpg', cv2.IMREAD_COLOR)

# check it has loaded

def light_leak():

    if not img is None:

        # Now we need to generate the light streak - randomly generating some y = mx + c
        # To start with I'll pick two x values in the central 1/3 of the image for the line to go between

        image_y, image_x, image_channels = img.shape

        point_1 = (0, random.randint(2*image_x//5, 3*image_x//5))
        point_2 = (image_y, random.randint(2*image_x//5, 3*image_x//5))

        m = sys.maxsize & 2 + 1
        if point_1[1] != point_2[1]:
            m = image_y/(point_1[1]-point_2[1])
        c = image_y - m*point_2[1]

        # Compare each pixel against the line

        width = 30
        attenuation = 5
        bright_factor = 1.7

        for y in range(image_y):
            for x in range(image_x):
                dist = abs(c + m*x - y)/math.sqrt(1 + math.pow(m, 2))
                if dist < width:
                    img.itemset((y, x, 0), min(img.item(y, x, 0)*bright_factor, 255))
                    img.itemset((y, x, 1), min(img.item(y, x, 1)*bright_factor, 255))
                    img.itemset((y, x, 2), min(img.item(y, x, 2)*bright_factor, 255))
                elif dist < width + attenuation:
                    att_factor = math.pow(bright_factor, 1 - 2*(dist-width)/attenuation)
                    img.itemset((y, x, 0), min(img.item(y, x, 0)*att_factor, 255))
                    img.itemset((y, x, 1), min(img.item(y, x, 1)*att_factor, 255))
                    img.itemset((y, x, 2), min(img.item(y, x, 2)*att_factor, 255))
                else:
                    img.itemset((y, x, 0), max(img.item(y, x, 0)/bright_factor, 0))
                    img.itemset((y, x, 1), max(img.item(y, x, 1)/bright_factor, 0))
                    img.itemset((y, x, 2), max(img.item(y, x, 2)/bright_factor, 0))


        cv2.imshow(windowName, img)
        key = cv2.waitKey(0)

        if (key == ord('x')):
            cv2.destroyAllWindows()

        
    else:
        print("No image file successfully loaded.")

light_leak()