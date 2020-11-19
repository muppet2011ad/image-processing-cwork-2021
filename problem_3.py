import numpy as np
import cv2, random, sys, math, cProfile

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

                    

    
def beauty_filter(img, colour_sigma=30, space_sigma=20, mode="warm"):

    image_y, image_x, image_channels = img.shape # Get dimensions of image to work with

    new_img = bilateralFilter(img, colour_sigma, space_sigma) # Apply bilaterial filtering with given parameters

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

# RECODE BILATERAL FILTER TO DO IT MYSELF