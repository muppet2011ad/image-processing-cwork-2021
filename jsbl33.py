import numpy as np
import cv2, random, sys, math, cProfile

### Problem 1 code following

def clip(val, minimum, maximum):
    return sorted((minimum, val, maximum))[1]

def problem_1(img, bright_factor = 2, blend_factor = 0.8, rainbow = False):
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

        return new_img



### Problem 2 code following

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

                    val += img.item(pixel_y,pixel_x)*mask.item(j,i) # Multiply the pixel by the mask and add it to the su
            new_img.itemset((y,x), int(clip(val, 0, 255))) # Copy the result of the sum into the output image
    return new_img

def pencilFilter(img, blend_factor): # Function to apply a pencil filter to a greyscale image (output image will also be greyscale)
    y_dim, x_dim = img.shape # Get dimensions of the shape

    noise = np.zeros(img.shape, dtype=np.uint8) # Generate empty image to use as noise mask

    for y in range(y_dim):
        for x in range(x_dim):
            noise.itemset((y, x), np.clip((1+np.random.normal(0, 0.1))*img.item(y, x), 0, 255)) # Generate random gaussian noise and merge with source

    # motion_blur_filter = np.array([[0, 0, 0, 0, 1/5],
    #                             [0, 0, 0, 1/5, 0],
    #                             [0, 0, 1/5, 0, 0],
    #                             [0, 1/5, 0, 0, 0],
    #                             [1/5, 0, 0, 0, 0]]) # Image filter for motion blur (just blurs along a line)

    motion_blur_filter = np.array([[0, 0, 0, 0, 0, 0, 1./14.],
                                   [0, 0, 0, 0, 0, 1./7., 0],
                                   [0, 0, 0, 0, 1./7., 0, 0],
                                   [0, 0, 0, 2./7., 0, 0, 0],
                                   [0, 0, 1./7., 0, 0, 0, 0],
                                   [0, 1./7., 0, 0, 0 ,0, 0],
                                   [1./14., 0, 0, 0, 0, 0, 0]], dtype=np.float64) # Image filter for motion blur (just blurs along a line)

    final_mask = cv2.filter2D(noise, 0, motion_blur_filter) # Convolves noisy image with motion blur to create effect

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

    return new_img


### Problem 3 code following

gaussians = {}

def gaussian(x, sigma=1, mu=0):
    if (x, sigma, mu) in gaussians:
        return gaussians[(x, sigma, mu)]
    else:
        value = (1/(sigma*((2*math.pi)**0.5)))*math.e**(-0.5*((x-mu)**2)/(sigma**2))
        gaussians[(x, sigma, mu)] = value
        return value

def bilateralFilter(img, colour_sigma, space_sigma):
    output = np.zeros(img.shape, np.uint8) # Create new image for output (we can't do this in place)
    y_dim = img.shape[0]
    x_dim = img.shape[1] # Get dimensions of image
    #d = max(colour_sigma, space_sigma)
    d = 3
    for y in range(y_dim):
        for x in range(x_dim): # Iterate through every pixel
            bilateral_numerator_BGR = [0, 0, 0]
            bilateral_denominator_BGR = [0, 0, 0] # Variables for numerator and denominator (per lecture 5 slide 31)
            pixel = [img.item(y, x, 0), img.item(y, x, 1), img.item(y, x, 2)] # Grab the pixel data here (no need to redo this for every neighbour)
            for j in range(y-d, y+d):
                for i in range(x-d, x+d): # Get neighbourhood
                    if j < 0: # Series of if statements "reflect" coords outside of the image back within the bounds of the image
                        j = -j
                    if i < 0:
                        i = -i
                    if j > y_dim-1:
                        j = 2*(y_dim-1) - j # Equivalent to j = y_dim - (j - y_dim)
                    if i > x_dim-1:
                        i = 2*(x_dim-1) - i
                    cart_dist = ((y-j)**2 + (x-i)**2)**0.5 # Calculate the cartesian distance between pixels (same for all channels)
                    gauss_dist = gaussian(cart_dist, space_sigma) # Get the output of the gaussian function for the distance component
                    for channel in range(3): # Handle each colour channel separately
                        coords = (j, i, channel)
                        weight = gauss_dist*gaussian(abs(img.item(coords)-pixel[channel]), colour_sigma) # Get the gaussian value for the intensity difference
                        bilateral_denominator_BGR[channel] += weight
                        bilateral_numerator_BGR[channel] += weight*img.item(coords) # Add to the summations on the top and bottom of the fraction for I_output
            for channel in range(3): # Loop to copy calculated data into the new image
                output.itemset((y, x, channel), bilateral_numerator_BGR[channel]//bilateral_denominator_BGR[channel])
    return output

def contrast_stretch(img, a=255, b=0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    c = np.amax(v)
    d = np.amin(v)
    for y in range(v.shape[0]):
        for x in range(v.shape[1]):
            v.itemset((y, x), (v.item(y, x)-c)*((a-b)/(c-d))+a)
    img_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


    
def problem_3(img, colour_sigma=20, space_sigma=10, mode="warm"):

    image_y, image_x, image_channels = img.shape # Get dimensions of image to work with

    new_img = bilateralFilter(img, colour_sigma, space_sigma) # Apply bilaterial filtering with given parameters

    new_img = contrast_stretch(new_img)

    red_adjust = {}
    blue_adjust = {} # Create lookup dictionaries for colour values


    if mode == "cold": # Same as warm, but drop red and increase blue
        red_pow = 0.96
        blue_pow = 1.04
        for i in range(256):
            red_adjust[i] = max(math.pow(i, red_pow), 0)
            blue_adjust[i] = min(math.pow(i, blue_pow), 255)
    else: # Generate colour values to "warm" image i.e. drop blue and increase red
        red_pow = 1.04
        blue_pow = 0.96 # Some powers for altering values
        for i in range(256): # For every possible red value
            red_adjust[i] = min(math.pow(i, red_pow), 255) # Raise to power and cap at 255
            blue_adjust[i] = max(math.pow(i, blue_pow), 0) # Raise to power and cap at 0

    for y in range(image_y):
        for x in range(image_x): # Iterate over the entire image
            new_img.itemset((y, x, 0), blue_adjust[new_img.item(y, x, 0)])
            new_img.itemset((y, x, 2), red_adjust[new_img.item(y, x, 2)]) # Replace the pixel value from the lookup table


    return new_img


### Problem 4 code following

def xy_to_polar(xy, centre):
    dx = xy[0] - centre[0]
    dy = xy[1] - centre[1]
    r = math.sqrt(dx**2 + dy**2)
    theta = math.atan2(dy, dx)
    return [r, theta]

def polar_to_xy(rtheta, centre):
    dx = rtheta[0] * math.cos(rtheta[1])
    dy = rtheta[0] * math.sin(rtheta[1])
    x = dx + centre[0]
    y = dy + centre[1]
    return [x, y]

def gaussian_low_pass(img, sigma):
    channels = cv2.split(img)
    for i in range(len(channels)):
        c = channels[i]
        c_fourier = np.fft.fftshift(np.fft.fft2(c))
        img_centre = (c_fourier.shape[0]//2, c_fourier.shape[1]//2)
        for y in range(c_fourier.shape[0]):
            for x in range(c_fourier.shape[1]):
                dx = x - img_centre[1]
                dy = y - img_centre[0]
                G = math.exp(-(dx**2 + dy**2)/2*(sigma**2))
                c_fourier.itemset((y, x), c_fourier.item(y, x)*G)
        channels[i] = np.uint8(np.clip(abs(np.fft.ifft2(np.fft.ifftshift(c_fourier))), 0, 255))
    return cv2.merge([channels[0], channels[1], channels[2]])

def subtract_image(img1, img2):
    return np.uint8(abs(np.subtract(img1, img2)))

def problem_4(img, angle=(3*math.pi/4), radius=150, interpolation="nn", prefilter=False):
    image_y, image_x, image_channels = img.shape
    output = np.zeros(img.shape, dtype=np.uint8)
    img_centre = (image_x//2, image_y//2)
    if prefilter:
        img = gaussian_low_pass(img, 0.0125)
    for y in range(image_y):
        for x in range(image_x):
            src_coords = [x,y]
            dist_from_centre = math.sqrt((x-img_centre[0])**2 + (y-img_centre[1])**2)
            if dist_from_centre < radius:
                as_polar = xy_to_polar((x, y), img_centre)
                as_polar[1] += angle*((1-(dist_from_centre/radius))**1.5)
                src_coords = polar_to_xy(as_polar, img_centre)
            for c in range(image_channels):
                if interpolation == "nn":
                    input_x = clip(round(src_coords[0]), 0, image_x-1)
                    input_y = clip(round(src_coords[1]), 0, image_y-1)
                    output.itemset((y, x, c), img.item(int(input_y), int(input_x), c))
                else:
                    x1 = int(clip(math.floor(src_coords[0]), 0, image_x-1))
                    y1 = int(clip(math.floor(src_coords[1]), 0, image_y-1))
                    x2 = int(clip(math.ceil(src_coords[0]), 0, image_x-1))
                    y2 = int(clip(math.ceil(src_coords[1]), 0, image_y-1))
                    input_x = clip(src_coords[0], 0, image_x-1)
                    input_y = clip(src_coords[1], 0, image_y-1)
                    if x2 != x1:
                        I_y1 = img.item(y1, x1, c)*(x2-input_x)/(x2-x1) + img.item(y1, x2, c)*(input_x-x1)/(x2-x1)
                        I_y2 = img.item(y2, x1, c)*(x2-input_x)/(x2-x1) + img.item(y2, x2, c)*(input_x-x1)/(x2-x1)
                    else:
                        I_y1 = img.item(y1, x1, c)
                        I_y2 = img.item(y2, x1, c)
                    if y2 != y1:
                        I = I_y1*(y2-input_y)/(y2-y1) + I_y2*(input_y-y1)/(y2-y1)
                    else:
                        I = I_y1
                    output.itemset((y, x, c), I)


    return output


if __name__ == "__main__":
    problem = int(sys.argv[1])
    output = np.zeros((1,1,3), np.uint8)
    if len(sys.argv) < 3:
        print("Specify image file!")
    args = sys.argv[2:]
    img = cv2.imread(args[0], cv2.IMREAD_COLOR)
    if problem == 1:
        if len(args) == 1:
            print("No filter params given, using defaults.")
            output = problem_1(img)
        elif len(args) == 2:
            print("Only rainbow param given, using defaults for others.")
            output = problem_1(img, rainbow=(args[1]=="rainbow"))
        elif len(args) < 4:
            print("Please give both brightness/blending parameters or none at all.")
            sys.exit()
        else:
            output = problem_1(img, float(args[2]), float(args[3]), (args[1]=="rainbow"))
    elif problem == 2:
        if len(args) == 1:
            print("No filter params given, using defaults.")
            output = problem_2(img)
        elif len(args) == 2:
            print("Only blend factor given, defaulting to b/w.")
            output = problem_2(img, float(args[1]))
        else:
            output = problem_2(img, float(args[1]), args[2] == "colour")
    elif problem == 3:
        if len(args) == 1:
            print("No filter params given, using defaults.")
            output = problem_3(img)
        elif len(args) == 2:
            print("Give both sigma values or neither.")
            sys.exit()
        elif len(args) == 3:
            print("No colour temp params given, defaulting to warm")
            output = problem_3(img, int(args[1]), int(args[2]))
        else:
            output = problem_3(img, int(args[1]), int(args[2]), args[3])
    elif problem == 4:
        if len(args) == 1:
            print("No filter params given, using defaults.")
            output = problem_4(img)
        elif len(args) == 2:
            print("Provide both swirl parameters or none at all")
            sys.exit()
        elif len(args) == 3:
            print("No interpolation/prefilter params given, defaulting to nearest neighbour and no prefiltering")
            output = problem_4(img, float(args[1]), int(args[2]))
        elif len(args) == 4:
            print("Defaulting to no prefiltering")
            output = problem_4(img, float(args[1]), int(args[2]), args[3])
        else:
            output = problem_4(img, float(args[1]), int(args[2]), args[3], args[4] == "prefilter")
    
    cv2.imshow("Output", output)
    key = cv2.waitKey(0)
    if key == ord('x'):
        cv2.destroyAllWindows()
        

