import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from skimage import data, img_as_float
from skimage import exposure

# Definition of constants
SIN_METHOD = 'sin_method'
FOURIER_METHOD = 'fourier_method'

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
    """ TODO documentation --> TUM method 1 (add reference to paper (?) 
    This function returns the function to be adjusted for the Fourier method
    params:
    x: Value of the pixel
    phase: Phase of the X-ray
    amp: amplitude of the X-ray
    mean: Mean value calculated 
    """	
    return mean + amp * np.cos(x + phase)	

def sin(x, a, b, c, d):
    """TODO documentation --> Method 2 added for comparison"""
    return a*np.sin(b*x + c) + d

def wrap_phase(inarray):	
    """TODO documentation"""
    outarray = np.angle(np.exp(1j * inarray))	
    return outarray

def Guess(curve):	
    """TODO documentation"""
    xvall=np.arange(len(curve))	
    ff=np.fft.fftfreq(len(xvall), (xvall[1]-xvall[0]))	
    Fyy = abs(np.fft.fft(np.array(curve)))	
    guess_freq=2*np.pi*abs(ff[np.argmax(Fyy[1:])+1])	
    return np.array([np.array(curve).std()*2**0.5, guess_freq, 0, np.array(curve).mean()])

def fit(ilum_curves, i, j, names, method,images=[]):
    """TODO documentation"""
    xdata = np.arange(len(names))
    ydata = ilum_curves[i,j]
    if method == SIN_METHOD:
        popt, _ = curve_fit(sin, xdata, ydata, p0=Guess(ydata), maxfev=2000)
        return popt
    elif method == FOURIER_METHOD:
        if images == []:
            raise Exception('The images array cannot be empty for the Fourier method')
        ft_ref_curve = np.fft.fft(ydata)	
        f0, f1 = ft_ref_curve[0], ft_ref_curve[1]	
        f0 = f0.real	
        mean_value = f0 / len(images)	
        amplitude = 2 * np.abs(f1) / len(images)	
        phase_shift = np.angle(f1)	
        return np.array([amplitude, phase_shift, mean_value])
    else:
        raise Exception('Method unrecognized.')

def padding(image,size):
    """TODO documentation"""	
    rows, columns = np.shape(image)[0], np.shape(image)[1]	
    padded_image = np.zeros((rows + 2*size, columns + 2*size))	
    #padded_image[size:size+rows, size:size+columns] = image	
    padded_image[size:-size, size:-size] = image	
    return padded_image	

def deppading(padded_image,size):	
    """TODO documentation"""
    return padded_image[size:-size, size:-size]	

def reduce_dim(image, factor):
    """TODO documentation"""	
    rows, columns = np.shape(image)[0], np.shape(image)[1]	
    new_image = np.zeros((int(rows/factor), int(columns/factor)))	
    for i in range(0,rows,factor):	
        for j in range(0,columns,factor):	
            new_image[int(i/factor),int(j/factor)] = np.mean(image[i:i+factor, j:j+factor])	
    return new_image

def image_preparation(image):	
    """TODO documentation"""
    rows, columns = np.shape(image)[0], np.shape(image)[1]	
    average = np.mean(image)	
    new_image = image
    row_criteria, column_criteria = (2.0/3.0)*rows, (2.0/3.0)*columns	
    for i in range(rows):	
        if( np.count_nonzero(image[i,:]) <  row_criteria):	
            new_image[i,:] = average*np.ones(rows)	
    	
    for j in range(columns):	
        if( np.count_nonzero(image[:,j]) < column_criteria ):	
            new_image[:,j] = average*np.ones(columns)	
    return new_image

def interpolation(image, factor=2):
    """TODO documentation"""
    # Creating x,y,z arrays
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
    """TODO documentation"""
    excess_cold = int((np.shape(kernel_cold)[0] - 1)/2)
    excess_hot = int((np.shape(kernel_hot)[0] - 1)/2)

    n_pixels_kernel_cold = np.shape(kernel_cold)[0]*np.shape(kernel_cold)[1]
    acceptable_pixels_cold = ((np.shape(kernel_cold)[0]-1)/np.shape(kernel_cold)[0])*n_pixels_kernel_cold
    n_pixels_kernel_hot = np.shape(kernel_hot)[0]*np.shape(kernel_hot)[1]
    acceptable_pixels_hot = ((np.shape(kernel_hot)[0]-1)/np.shape(kernel_hot)[0])*n_pixels_kernel_hot

    mean_image, std_image = np.mean(image), np.std(image)

    minimum_acceptable = mean_image - stds_cold*std_image
    maximum_acceptable = mean_image + stds_hot*std_image
    logger.info("Minimum: ", minimum_acceptable)
    logger.info("Maximum: ", maximum_acceptable)

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
                #elif(result[i,j] <= minimum_acceptable and np.count_nonzero(padded_image_cold[i-excess_cold:i+excess_cold+1, j-excess_cold:j+excess_cold+1]) <= acceptable_pixels_cold):
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
                #elif(result2[i,j] >= maximum_acceptable and np.count_nonzero(padded_image_hot[i-excess_hot:i+excess_hot+1, j-excess_hot:j+excess_hot+1]) <= acceptable_pixels_hot):
                    imagen_interp = interpolation(padded_image_hot[i-excess_hot:i+excess_hot+1, j-excess_hot:j+excess_hot+1])
                    multiplication = imagen_interp*kernel_hot
                    result2[i,j] = np.sum(multiplication)

        result2 = deppading(result2,excess_hot)
        return result2
