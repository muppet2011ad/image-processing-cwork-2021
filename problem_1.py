import numpy as np
import cv2, random, sys, math, cProfile

# check it has loaded

def clip(val, minimum, maximum):
    return sorted((minimum, val, maximum))[1]

def problem1(img, bright_factor = 2, blend_factor = 0.8, rainbow = False):
    if not img is None:

        windowName = "Light Leak Filter"

        # Now we need to generate the light streak - randomly generating some y = mx + c
        # To start with I'll pick two x values in the central 1/3 of the image for the line to go between

        image_y, image_x, image_channels = img.shape # Get the dimensions of the image

        point_1 = (0, random.randint(2*image_x//5, 3*image_x//5))
        point_2 = (image_y, random.randint(2*image_x//5, 3*image_x//5)) # Pick two random points at the top and bottom of the image

        m = sys.maxsize & 2 + 1
        if point_1[1] != point_2[1]:
            m = image_y/(point_1[1]-point_2[1])
        c = image_y - m*point_2[1] # Work out the equation of the line between them

        # Compare each pixel against the line

        width = 30
        attenuation = 5 # Good values for making this work
        max_hue = 160

        mask = np.zeros(img.shape, np.uint8) # Generate a blank image to act as a mask

        for y in range(image_y):
            for x in range(image_x): # Iterate through the image
                dist = abs(c + m*x - y)/math.sqrt(1 + math.pow(m, 2)) # Calculate the distance of between the point (x,y) and the line
                img.itemset((y, x, 0), max(img.item(y, x, 0)/bright_factor, 0))
                img.itemset((y, x, 1), max(img.item(y, x, 1)/bright_factor, 0))
                img.itemset((y, x, 2), max(img.item(y, x, 2)/bright_factor, 0)) # Darken it accordingly
                if dist < width: # If the resulting distance is less than the width of the beam
                    new_bright_factor = bright_factor # Get ready to adjust the brightness factor depending on position in the beam
                    if not rainbow:
                        if abs(dist) < width/1.5:
                            new_bright_factor *= 2-(1.5*abs(dist)/width) # This makes the centre of the beam brighter
                        mask.itemset((y, x, 0), min(img.item(y, x, 0)*new_bright_factor, 255))
                        mask.itemset((y, x, 1), min(img.item(y, x, 1)*new_bright_factor, 255))
                        mask.itemset((y, x, 2), min(img.item(y, x, 2)*new_bright_factor, 255)) # Copy brightness data into mask, capping at 255
                    else: # If we're generating a rainbow, it's a bit more complicated
                        sat_factor = 1
                        if abs(dist) < width/3: # We use a narrower bright streak for this
                            new_bright_factor *= 2-(3*abs(dist)/width)
                            sat_factor = 0.1+0.9*(3*abs(dist)/width)
                        if m*x - y + c < 0:
                            dist = -dist # Work out which side of the line we're on (important for working out hue)
                        pix_bgr = np.uint8([[[img.item(y, x, 0), img.item(y, x, 1), img.item(y, x , 2)]]]) # Get the value of the pixel
                        pix_hsv = cv2.cvtColor(pix_bgr, cv2.COLOR_BGR2HSV) # Convert to HSV (rainbows are easier to make by varying hue)
                        pix_hsv.itemset((0, 0, 0), ((dist+width)/(2*width))*max_hue) # Calculate value for hue
                        pix_hsv.itemset((0, 0, 1), sat_factor*max(150, pix_hsv.item(0, 0, 1))) # Saturation should be at least 150, but also no less than the existing saturation
                        pix_hsv.itemset((0, 0, 2), clip(pix_hsv.item(0, 0, 2)*new_bright_factor, 50, 255)) # Apply brightening effect
                        pix_bgr = cv2.cvtColor(pix_hsv, cv2.COLOR_HSV2BGR) # Convert back to BGR
                        mask.itemset((y, x, 0), pix_bgr.item(0, 0, 0))
                        mask.itemset((y, x, 1), pix_bgr.item(0, 0, 1))
                        mask.itemset((y, x, 2), pix_bgr.item(0, 0, 2)) # Copy pixel data into mask
                elif dist < width + attenuation: # Smooth transition into normal image makes it look more natural
                    att_factor = math.pow(bright_factor, 1 - 2*(dist-width)/attenuation) # Function to transition from bright to dark as we move further out
                    if not rainbow:
                        mask.itemset((y, x, 0), min(img.item(y, x, 0)*att_factor, 255))
                        mask.itemset((y, x, 1), min(img.item(y, x, 1)*att_factor, 255))
                        mask.itemset((y, x, 2), min(img.item(y, x, 2)*att_factor, 255)) # Copy image data to mask
                    else:
                        hue = 0 # Min hue at one end
                        if m*x - y + c > 0:
                            hue = max_hue # Otherwise max hue
                        pix_bgr = np.uint8([[[img.item(y, x, 0), img.item(y, x, 1), img.item(y, x , 2)]]]) # Get pixel data from image
                        pix_hsv = pix_hsv = cv2.cvtColor(pix_bgr, cv2.COLOR_BGR2HSV) # Convert to hsv
                        pix_hsv.itemset((0, 0, 0), hue) # Set hue
                        pix_hsv.itemset((0, 0, 1), max((150/bright_factor)*att_factor, pix_hsv.item(0, 0, 1))) # Saturation decided as before but also attenuating
                        pix_hsv.itemset((0 ,0 ,2), pix_hsv.item(0, 0, 2)*att_factor) # Brightness also attenuates
                        pix_bgr = cv2.cvtColor(pix_hsv, cv2.COLOR_HSV2BGR) # Convert back to BGR
                        mask.itemset((y, x, 0), pix_bgr.item(0, 0, 0))
                        mask.itemset((y, x, 1), pix_bgr.item(0, 0, 1))
                        mask.itemset((y, x, 2), pix_bgr.item(0, 0, 2)) # Copy pixel data into mask
                else:
                    mask.itemset((y, x, 0), max(img.item(y, x, 0)/bright_factor, 0))
                    mask.itemset((y, x, 1), max(img.item(y, x, 1)/bright_factor, 0))
                    mask.itemset((y, x, 2), max(img.item(y, x, 2)/bright_factor, 0)) # Otherwise continue to copy image data through

        #img = cv2.addWeighted(img, 1-blend_factor, mask, blend_factor, 0) # Blend the two images together

        new_img = np.zeros(img.shape, np.uint8)
        for y in range(image_y):
            for x in range(image_x):
                new_img.itemset((y, x, 0), min(255, img.item(y, x, 0)*(1-blend_factor) + mask.item(y, x, 0)*blend_factor))
                new_img.itemset((y, x, 1), min(255, img.item(y, x, 1)*(1-blend_factor) + mask.item(y, x, 1)*blend_factor))
                new_img.itemset((y, x, 2), min(255, img.item(y, x, 2)*(1-blend_factor) + mask.item(y, x, 2)*blend_factor))

        cv2.imshow(windowName, new_img)
        key = cv2.waitKey(0)

        if (key == ord('x')):
            cv2.destroyAllWindows()

        
    else:
        print("No image file successfully loaded.")


args = sys.argv
if len(args) < 2:
    print("Specify image file!")
elif len(args) == 2:
    print("No filter params given, using defaults.")
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    problem1(img)
elif len(args) == 3:
    print("Only rainbow param given, using defaults for others.")
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    rainbow = False
    if args[2] == "rainbow":
        rainbow = True
    problem1(img, rainbow=rainbow)
elif len(args) < 5:
    print("Give both brightness/blending parameters or none at all.")
else:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    rainbow = False
    if args[2] == "rainbow":
        rainbow = True
    problem1(img, float(args[3]), float(args[4]), rainbow)
    