import numpy as np
import cv2, random, sys, math

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

def clip(val, minimum, maximum):
    return sorted((minimum, val, maximum))[1]

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


def swirlFilter(img, angle, radius, interpolation="nn"):
    image_y, image_x, image_channels = img.shape
    output = np.zeros(img.shape, dtype=np.uint8)
    img_centre = (image_x//2, image_y//2)
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
                    output.itemset((y, x, c), img.item(input_y, input_x, c))
                else:
                    x1 = clip(math.floor(src_coords[0]), 0, image_x-1)
                    y1 = clip(math.floor(src_coords[1]), 0, image_y-1)
                    x2 = clip(math.ceil(src_coords[0]), 0, image_x-1)
                    y2 = clip(math.ceil(src_coords[1]), 0, image_y-1)
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

test_img = cv2.imread("input_2.jpg", cv2.IMREAD_COLOR)
out_img = swirlFilter(test_img, 3*math.pi/4, 150, "bi")

cv2.imshow("Swirl Filter", out_img) # Show image
key = cv2.waitKey(0)

if (key == ord('x')):
    cv2.destroyAllWindows()