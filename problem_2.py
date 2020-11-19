import numpy as np
import cv2, random, sys, math, cProfile

def applyMask(img, mask): # Function to apply square mask to a greyscale image
    image_y, image_x = img.shape
    mask_y, mask_x = mask.shape # Get image and mask dimensions
    new_img = np.zeros(img.shape, dtype=np.uint8) # Create array for output image
    for y in range(image_y):
        for x in range(image_x): # Iterate through image pixels
            val = 0 # Variable to store sum as we go through the neighbourhood
            for j in range(mask_y):
                for i in range(mask_x): # Iterate through the size of the mask
                    offset = mask_y//2 # Calculate how far we are from the centre of the mask
                    pixel_y = y + j - offset
                    pixel_x = x + i - offset # Convert that into pixel coords

                    if pixel_y < 0:
                        pixel_y = -pixel_y
                    elif pixel_y > image_y - 1:
                        pixel_y = 2*(image_y-1) - pixel_y
                    if pixel_x < 0:
                        pixel_x = -pixel_x
                    elif pixel_x > image_x -1:
                        pixel_x = 2*(image_x-1) - pixel_x # If the pixel falls outside the image, "reflect" it back into valid coords

                    val += img.item(pixel_y,pixel_x)*mask.item(j,i) # Multiply the pixel by the mask and add it to the sum
            new_img.itemset((y,x), val) # Copy the result of the sum into the output image
    return new_img

def pencilFilter(img, blend_factor): # Function to apply a pencil filter to a greyscale image (output image will also be greyscale)
    y_dim, x_dim = img.shape # Get dimensions of the shape

    noise = np.zeros(img.shape, dtype=np.uint8) # Generate empty image to use as noise mask

    for y in range(y_dim):
        for x in range(x_dim):
            noise.itemset((y, x), np.clip((1+np.random.normal(0, 0.1))*img.item(y, x), 0, 255)) # Generate random gaussian noise and merge with source

    motion_blur_filter = np.array([[0, 0, 0, 0, 1/5],
                                [0, 0, 0, 1/5, 0],
                                [0, 0, 1/5, 0, 0],
                                [0, 1/5, 0, 0, 0],
                                [1/5, 0, 0, 0, 0]]) # Image filter for motion blur (just blurs along a line)

    final_mask = applyMask(noise, motion_blur_filter) # Convolves noisy image with motion blur to create effect

    new_img = np.zeros(img.shape, np.uint8) # Create new output image
    for y in range(y_dim):
        for x in range(x_dim): # Iterate through dimensions of source image
            new_img.itemset((y, x), min(255, img.item(y, x)*(1-blend_factor) + final_mask.item(y, x)*blend_factor))
            # Blend the images based on a blending factor

    return new_img

def problem_2(img, blend_factor=0.6, colour=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y_dim, x_dim = img.shape

    if colour:
        channels = random.sample([0, 1, 2], 2)
        channel_1 = pencilFilter(img, blend_factor)
        channel_2 = pencilFilter(img, blend_factor)
        imgs = [channel_1, channel_2]
        new_img = np.zeros(img.shape + (3,), np.uint8)
        for y in range(y_dim):
            for x in range(x_dim):
                for c in range(2):
                    new_img.itemset((y, x, channels[c]), imgs[c].item(y, x))
    else:
        new_img = pencilFilter(img, blend_factor)

    windowName = "Pencil Filter"
    cv2.imshow(windowName, new_img)
    key = cv2.waitKey(0)

    if (key == ord('x')):
        cv2.destroyAllWindows()

args = sys.argv

if len(args) < 2:
    print("Specify image file!")
elif len(args) == 2:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    problem_2(img)
elif len(args) == 3:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    problem_2(img, float(args[2]))
else:
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    colour = args[3] == "colour"
    problem_2(img, float(args[2]), colour)
