# KARAGKIOZI EFKLEIA
# Computer Vision project

#libraries
import numpy as np
import cv2


##########  BASIC IMAGE FUNCTIONS

def myColorToGray(image):
    """convert a color image (RGB ) to grayscale using luminance weights"""
    gray_value = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    gray_img = gray_value.astype(np.uint8)
    return gray_img


def myConv2(image, kernel):
    """perform 2D convolution"""
    # Flipping the kernel
    kernel = np.flipud(np.fliplr(kernel))  # flip kernel for convolution
    k = kernel.shape[0]
    pad_width = k // 2

    #zero padding depending on the kernel size
    img_padded = np.pad(image, pad_width, mode='constant', constant_values=0)

    #output image initialized to zeros (same shape as input)
    output_img = np.zeros(image.shape)


    for x in range(pad_width, img_padded.shape[0] - pad_width):
        for y in range(pad_width, img_padded.shape[1] - pad_width):
            region = img_padded[x - pad_width:x + pad_width + 1, y - pad_width:y + pad_width + 1]
            output_img[x - pad_width, y - pad_width] = np.sum(region * kernel)

    return output_img


###########  NOISE ADDITION

def myImNoise(A, noise):
    """add Gaussian or salt-and-pepper noise to an image"""
    image = A.copy()

    if noise == "gaussian":
        mean, var = 0, 0.01
        sigma = np.sqrt(var)
        noise_matrix = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.add(image, noise_matrix)
        return noisy_image
    elif noise == "saltandpepper":
        pepper = 0.02   #probability
        salt = 1 - pepper

        noisy_image = image.copy()
        r = np.random.random(image.shape)
        noisy_image[r < pepper] = 0
        noisy_image[r > salt] = 1
        return noisy_image

    else:
        raise ValueError("noise must be 'gaussian' or 'saltandpepper'")


##############  FILTERS

def myImFilter(A, filtr, window_size):
    """Apply mean or median filter to an image."""
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be an odd integer greater than 1")

    image = A.copy()

    if filtr == 'mean':
        # kernel size (bigger size -> more blurr)
        kernel = np.ones((window_size, window_size), dtype=float)
        kernel /= kernel.sum()
        return myConv2(image, kernel)

    elif filtr == 'median':
        # zero padding depending on the window_size
        pad_width = window_size // 2
        img_padded = np.pad(image, pad_width, constant_values=(0))
        output = np.zeros_like(image)

        for i in range(pad_width, img_padded.shape[0] - pad_width):
            for j in range(pad_width, img_padded.shape[1] - pad_width):
                # Get the neighborhood around the current pixel
                neighborhood = img_padded[i - pad_width : i + pad_width + 1, j - pad_width: j + pad_width + 1]
                output[i - pad_width, j - pad_width] = np.median(neighborhood)
        return output

    else:
        raise ValueError("Unsupported filter type. Use 'mean' or 'median'.")


#############  EDGE DETECTION

def myEdgeDetection(A, method):
    """Perform Sobel, Prewitt, or Laplacian edge detection."""
    image = A.copy()

    if method == "sobel":
        fx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        fy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    elif method == "prewitt":
        fx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        fy = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

    elif method == "laplacian":
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        return myConv2(image, kernel)

    else:
        raise ValueError("Unsupported edge detection method.")

    gx = myConv2(image, fx)
    gy = myConv2(image, fy)
    
    #norm of gradient vector  magnitude
    magnitude = np.sqrt(gy ** 2 + gx ** 2)

    return magnitude


############  HELPER FUNCTION

def show(img, title="Image"):
    """Display an image and wait for key press."""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


########  MAIN SCRIPT

def main():
    # read image
    img = cv2.imread("Resources/Moonrise_Kingdom.png")
    #show(img, "Original")

    # get image dimensions
    h, w, d = img.shape
    print(f"Image dimensions width={w}, height={h}, depth={d}")

    # convert to grayscale
    A = myColorToGray(img)
    #show(A, "GrayScale")

    # scale to [0, 1]
    A = A / 255.0

    ## Gaussian noise
    B = myImNoise(A, noise="gaussian")
    ##show(B, "Gaussian Noise")
#
    ## salt & pepper noise
    B = myImNoise(A, noise="saltandpepper")
    ##show(B, "Salt & Pepper Noise")
#
    ## mean filter
    #C = myImFilter(B, filtr="mean", window_size=3)
    #show(C, "Mean Filter")

    # median filter
    C = myImFilter(B, filtr="median", window_size=3)
    show(C, "Median Filter")
    output_filename = "median.png"
    cv2.imwrite(output_filename, (C * 255).astype(np.uint8))  # Save as 8-bit image

    ## sobel edge detection
    #A_sobel = myEdgeDetection(A, method="sobel")
    #show(A_sobel, "Sobel")
#
    ## prewitt edge detection
    #A_prewitt = myEdgeDetection(A, method="prewitt")
    #show(A_prewitt, "Prewitt")

    ## laplacian edge detection
    #A_laplacian = myEdgeDetection(A, method="laplacian")
    #show(A_laplacian, "Laplacian")
    #
    ###### Some extra examples
    ##1
    ## Mean filter (window 9)
    #E = myImFilter(A, filtr="mean", window_size=9)
    #show(E, "Mean Filter (9x9)")
#
    ## Edge detection after heavy smoothing
    #show(myEdgeDetection(E, "sobel"), "Sobel after 9x9 Mean")
    #show(myEdgeDetection(E, "prewitt"), "Prewitt after 9x9 Mean")
    #show(myEdgeDetection(E, "laplacian"), "Laplacian after 9x9 Mean")
#    
    ##2
    ## Mean filter (window 3)
    #F = myImFilter(A, filtr="mean", window_size=3)
    #show(F, "Mean Filter (3x3)")
#
    ## Edge detection after light smoothing
    #show(myEdgeDetection(F, "sobel"), "Sobel after 3x3 Mean")
    #show(myEdgeDetection(F, "prewitt"), "Prewitt after 3x3 Mean")
    #show(myEdgeDetection(F, "laplacian"), "Laplacian after 3x3 Mean")




if __name__ == "__main__":
    main()
