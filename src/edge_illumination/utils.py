import logging
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

# Definition of constants
SIN_METHOD = 'sin_method'
FOURIER_METHOD = 'fourier_method'
INTENSITIES = 'intensities'

# Disabling library logs
from ei_logger import disable_loggers
disable_loggers(__name__)

# Setting the logger
logger = logging.getLogger(__name__)

# Initial parameters
n_pixels = 256	
pixel_size = 55/1000 # In mm	
colormap = "bone"

# Defining 3x3 matrices with entries 1/9
kernel_cold = np.ones((3,3))/(3*3)	
kernel_hot = np.ones((3,3))/(3*3)	

def func_fourier(x, amp, phase, mean):
    """
    Creates the function to be adjusted for the Fourier method to the edge illumination curve
    params:
    x: Value of each of the points of the edge illumination curve
    phase: Phase to be calculated for the edge illumination curve
    amp: Amplitude to be calculated for the edge illumination curve
    mean: Mean average of cosine function, represents the shift of the edge illumination curve
    returns:
    cosine function to be adjusted
    """	
    return mean + amp * np.cos(x + phase)	

def sin(x, a, b, c, d):
    """
    Defines a sine function that will be used for fitting the edge illumination curve
    params:
    a: Amplitude to be calculated for the edge illumination curve
    b: Factor that is multiplied by each of the points of the edge illumination curve
    c: Phase to be calculated for the edge illumination curve
    d: Mean average of sine function, represents the shift of the edge illumination curve
    returns:
    sine function to be adjusted
    """
    return a*np.sin(b*x + c) + d

def wrap_phase(inarray):	
    """
    Scales phase image between -pi and pi
    params:
    inarray: Phase image to be scaled within -pi and pi
    returns:
    outarray: Phase image scaled between -pi and pi
    """
    outarray = np.angle(np.exp(1j * inarray))	
    return outarray

def initial_estimation(curve):	
    """
    Function to be used in the sine method to estimate initial values for parameters.
    Parameters are calculated in the following way:
    a --> (2*std_dev_of_curve) **1/2
    b --> Frequency of dirac delta after Fourier transformation
    c --> Initial phase is setup to 0
    d --> Average of the curve --> vertical shift
    params: 
    curve: the curve from which the parameters will be estimated
    return:
    array with the initial estimation for the [a,b,c,d] parameters
    """
    xvall=np.arange(len(curve))	
    ff=np.fft.fftfreq(len(xvall), (xvall[1]-xvall[0]))	
    Fyy = abs(np.fft.fft(np.array(curve)))	
    guess_freq=2*np.pi*abs(ff[np.argmax(Fyy[1:])+1])
    return np.array([np.array(curve).std()*2**0.5, guess_freq, 0, np.array(curve).mean()])

def fit(ilum_curves, i, j, names, method,images=[]):
    """
    Fits data to a given function based on the selected method.
    params:
    ilum_curves = Edge illumination curves
    i: X position in an image
    j: Y position on an image
    names: Names given to the xdata
    method: Can be either sine or fourier method 
    images: array of images, to normalize data for Fourier method. (May be empty for sine method).
    returns:
    Optimal parameters calculated for either of the methods
    """
    xdata = np.arange(len(names))
    ydata = ilum_curves[i,j]
    if method == SIN_METHOD:
        logger.info('Calculating the optimal parameters for the sine method')
        popt, _ = curve_fit(sin, xdata, ydata, p0=initial_estimation(ydata), maxfev=2000)
        return popt
    elif method == FOURIER_METHOD:
        if images == []:
            raise Exception('The images array cannot be empty for the Fourier method')
        logger.info('Calculating FFT for ydata in the Fourier method')
        ft_ref_curve = np.fft.fft(ydata)
        # f0 represents the avg of the curve
        # f1 contains information of the dirac delta peak 	
        f0, f1 = ft_ref_curve[0], ft_ref_curve[1]	
        # Taking the real part of the avg of the curve
        f0 = f0.real
        logger.info('Calculating mean, amplitude and phase shift parameters')
        # Normalized to the number of images	
        mean_value = f0 / len(images)	
        # Amplitude is the real part of the peak, normalized to the number of images
        amplitude = 2 * np.abs(f1) / len(images)	
        phase_shift = np.angle(f1)	
        logger.debug(f'Mean value: {mean_value}')
        logger.debug(f'Amplitude: {amplitude}')
        logger.debug(f'Phase shift: {phase_shift}')
        return np.array([amplitude, phase_shift, mean_value])
    else:
        raise Exception('Method unrecognized.')

def padding(image,size):
    """
    Adds a padding of zeros with double a given size to an image. 
    params:
    image: Image to which the padding will be added
    size: Size of the padding that will be added
    returns:
    padded_image: Image with the padding that was previously added
    """	
    logger.info(f'Adding padding of size {size} to the images')
    rows, columns = np.shape(image)[0], np.shape(image)[1]	
    padded_image = np.zeros((rows + 2*size, columns + 2*size))	
    padded_image[size:-size, size:-size] = image	
    return padded_image	

def deppading(padded_image,size):	
    """
    Removes the padding from a padded image
    params:
    padded_image: Padded image to which the padding will be removed
    size: Size of the padding that was applied to the image
    returns:
    Image with the padding removed
    """
    logger.info(f'Removing padding of size {size} to the images')
    return padded_image[size:-size, size:-size]	

def reduce_dim(image, factor):
    """
    Reduces the dimension of an image by a given factor
    params:
    image: Image to be reduced 
    factor: Factor of reduction for the image
    returns:
    new_image: Reduced image
    """	
    logger.info(f'Reducing dimension of image by a factor of {factor}')
    rows, columns = np.shape(image)[0], np.shape(image)[1]	
    new_image = np.zeros((int(rows/factor), int(columns/factor)))	
    for i in range(0,rows,factor):	
        for j in range(0,columns,factor):	
            new_image[int(i/factor),int(j/factor)] = np.mean(image[i:i+factor, j:j+factor])	
    return new_image

def image_preparation(image, criteria=(2.0/3.0)):	
    """
    Prepares an image for processing. All rows/columns in which less than 2/3 (or given criteria) of pixels 
    are corrected are removed by the average of the image.
    params:
    image: Image to be prepared
    criteria: Number of pixels in rows/columns that are acceptable as a minimum threshold for non-zero pixels
    returns:
    new_image: Image corrected, with rows and columns that do not satisfy criteria, averaged to the mean
    """
    rows, columns = np.shape(image)[0], np.shape(image)[1]	
    average = np.mean(image)	
    new_image = image
    row_criteria, column_criteria = criteria*rows, criteria*columns	
    logger.info('Correcting rows in which non-zero pixels are more than 2/3 of its totality')
    for i in range(rows):	
        if( np.count_nonzero(image[i,:]) <  row_criteria):	
            new_image[i,:] = average*np.ones(rows)	
    logger.info('Correcting columns in which non-zero pixels are more than 2/3 of its totality')
    for j in range(columns):	
        if( np.count_nonzero(image[:,j]) < column_criteria ):	
            new_image[:,j] = average*np.ones(columns)	
    return new_image

def interpolation(image, factor=2):
    """
    Performs interpolation to an image, using the given factor to expand the image
    params:
    image: Image to be interpolated with a given factor
    factor: Factor for interpolation, which augments the image 
    returns
    reduced_dim_image: Image with applied interpolation and dimension subsequently reduced.
    """
    # Creating x,y,z arrays
    # z is the intensity value of the pixel
    x, y, z = [], [], [] 
    matrix_shape = np.shape(image)
    rows, columns = matrix_shape[0], matrix_shape[1]
    # Obtaining x,y,z data
    for i in range(rows):
        for j in range(columns):
            x.append(i)
            y.append(j)
            z.append(image[i,j])
    x, y, z = np.array(x), np.array(y), np.array(z)
    # Interpolating x,y,z data
    # Creating the grid. 
    grid_x, grid_y = np.mgrid[0:rows-1:factor*rows*1j, 0:columns-1:factor*columns*1j]  
    xy_data = np.vstack((x,y)).T 
    # Cubic interpolation
    new_image = griddata(xy_data, z, (grid_x, grid_y), method='cubic') 
    # Reducing the dimension of the image
    reduced_dim_image = reduce_dim(new_image, factor)
    return reduced_dim_image

def convolution(image,kernel_cold, kernel_hot, selective=False, stds_cold=1, stds_hot=1):
    """
    This method applies convolution to an image. First, uses the image_preparation method to apply preliminary
    corrections. Subsequently, applies convolutions to the image based on the given parameters
    params: 
    image: Image to be convoluted 
    kernel_cold: Kernel that will be applied to cold pixels (with a value of 0 or near 0). 
    kernel_hot: Kernel that will be applied to hot (saturated) pixels.
    selective: If True, the convolutions will be only added to damaged pixels of the image. If False, the convolutions
                will be applied to the whole image. 
    stds_cold: Standard deviation for cold pixels.
    stds_hot: Standard deviation for hot pixels
    returns:
    result/result2: Image with the convolutions applied, depending on the value of the selective parameter
    """
    # Applying preliminary corrections to the image
    image = image_preparation(image)
    # Determining the number of excessive cold and hot pixels
    excess_cold = int((np.shape(kernel_cold)[0] - 1)/2)
    excess_hot = int((np.shape(kernel_hot)[0] - 1)/2)

    # Calculating the number of kernel pixels, and acceptable cold/hot pixels.
    n_pixels_kernel_cold = np.shape(kernel_cold)[0]*np.shape(kernel_cold)[1]
    acceptable_pixels_cold = ((np.shape(kernel_cold)[0]-1)/np.shape(kernel_cold)[0])*n_pixels_kernel_cold
    n_pixels_kernel_hot = np.shape(kernel_hot)[0]*np.shape(kernel_hot)[1]
    acceptable_pixels_hot = ((np.shape(kernel_hot)[0]-1)/np.shape(kernel_hot)[0])*n_pixels_kernel_hot

    mean_image, std_image = np.mean(image), np.std(image)

    # Determining the minimum/maximum values per pixel, based on cold/hot pixels
    minimum_acceptable = mean_image - stds_cold*std_image
    maximum_acceptable = mean_image + stds_hot*std_image
    logger.info("Minimum value for pixel: ", minimum_acceptable)
    logger.info("Maximum value for pixel: ", maximum_acceptable)

    # Applies convolutions to the whole image
    if not selective:
        padded_image = padding(image,excess_cold)
        padded_rows, padded_columns = np.shape(padded_image)[0], np.shape(padded_image)[1]

        result = np.zeros( (padded_rows, padded_columns) )
        for i in range(excess_cold, padded_rows-excess_cold):
            for j in range(excess_cold, padded_columns-excess_cold):
                if(np.count_nonzero(padded_image[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1]) > acceptable_pixels_cold):
                    multiplication = padded_image[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1]*kernel_cold
                    result[i,j] = np.sum(multiplication)
                elif(np.count_nonzero(padded_image[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1]) <= acceptable_pixels_cold):
                    imagen_interp = interpolation(padded_image[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1])
                    multiplication = imagen_interp*kernel_cold
                    result[i,j] = np.sum(multiplication)
        result = deppading(result,excess_cold)
        return result
    # Applies convolutions to the damaged pixels
    else:
        padded_image_cold = padding(image,excess_cold)
        padded_rows_cold, padded_columns_cold = np.shape(padded_image_cold)[0], np.shape(padded_image_cold)[1]

        result = padded_image_cold
        for i in range(excess_cold, padded_rows_cold-excess_cold):
            for j in range(excess_cold, padded_columns_cold-excess_cold):
                if(result[i,j] <= minimum_acceptable and np.count_nonzero(padded_image_cold[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1]) > acceptable_pixels_cold):
                    multiplication = padded_image_cold[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1]*kernel_cold
                    result[i,j] = np.sum(multiplication)
                else:
                    imagen_interp = interpolation(padded_image_cold[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1])
                    multiplication = imagen_interp*kernel_cold
                    result[i,j] = np.sum(multiplication)

        result = deppading(result,excess_cold)
        padded_image_hot = padding(result,excess_hot)
        padded_rows_hot, padded_columns_hot = np.shape(padded_image_hot)[0], np.shape(padded_image_hot)[1]
        result2 = padded_image_hot

        for i in range(excess_hot, padded_rows_hot-excess_hot):
            for j in range(excess_hot, padded_columns_hot-excess_hot):
                if(result2[i,j] >= maximum_acceptable and np.count_nonzero(padded_image_hot[i-excess_hot:i+excess_hot+1, j-excess_hot:j+excess_hot+1]) > acceptable_pixels_hot):
                    multiplication = padded_image_hot[i-excess_hot:i+excess_hot+1, j-excess_hot:j+excess_hot+1]*kernel_hot
                    result2[i,j] = np.sum(multiplication)
                else:
                    imagen_interp = interpolation(padded_image_hot[i-excess_hot:i+excess_hot+1, j-excess_hot:j+excess_hot+1])
                    multiplication = imagen_interp*kernel_hot
                    result2[i,j] = np.sum(multiplication)
        result2 = deppading(result2,excess_hot)
        return result2

def profile_pixel(images,i,j):
    """
    Returns the pixel profile in the i,j position for all images
    params:
    images: Set of images to be profiled
    i: X position of pixel
    j: Y position of pixel
    returns:
    array of points in the [i,j] for all images
    """
    points = []
    for image in images:
        points.append(image[i,j])
    return points

def correct_dead_pixels(images, dim_x=256, dim_y=256):
    """
    Corrects dead pixels in the image and sets them as the mean of the image
    params:
    """
    for img in images: 
        mean = np.mean(img)
        for i in range(dim_x):
            for j in range(dim_y):
                if img[i,j] == 0:
                    img[i,j] = mean
