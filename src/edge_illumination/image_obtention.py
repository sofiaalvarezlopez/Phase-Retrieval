import enum
import logging
import os
import numpy as np
from utils import *
from ei_logger import *
from skimage import exposure
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from skimage import data, img_as_float

# Definition of the arguments for the parser
ap = ArgumentParser(description = 'Edge Illumination: image generation', 
            epilog='This scipt generates attenuation, visibility and phase contrast images for the edge illumination method')
ap.add_argument('--root-dir',dest='root_dir', help='root directory with all the data', required=True)
ap.add_argument('--ff-dir',dest='ff_dir', default='2. FF/', help='path to data with the Flat Field data',required=True)
ap.add_argument('--raw-dir',dest='raw_dir', default='1. RAW/', help='path to data with the Raw image data',required=True)
ap.add_argument('--output-dir',dest='output_dir', default='./output/', help='path to output directory',required=True)
ap.add_argument('--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                default='DEBUG', help='Set log level (default DEBUG)')
ap.add_argument('--save-aux-plots', dest='save_aux_plots', action='store_const', const=True, default=False, 
                help='If the plots for grids and illumination curves should be stored')
ap.add_argument('--method', dest='method', choices=[SIN_METHOD, FOURIER_METHOD, INTENSITIES],
                required=True, help='Method to be used for the image generation')
ap.add_argument('--dead-pixel-method', dest='dead_pixel_method', choices=[MEAN, CONVOLUTIONS], default=MEAN,
                required=True, help='Method to be used for dead pixel correction')
ap.add_argument('--selective', dest='selective', action='store_const', const=True, default=False,
                help=f'If {CONVOLUTIONS} method for dead pixel correction is chosen, indicates if convolutions should be \
                    only applied to damaged pixels (True) or to the whole image (False)')
args = ap.parse_args()

# Setting the logger
set_args(args)
logger = get_logger(__name__)

# Getting parser parameters
root_dir = args.root_dir
ff_dir = args.ff_dir
raw_dir = args.raw_dir
output_dir = args.output_dir

if args.method == INTENSITIES:
    print('Hola')
else:
    # Defining subdirectories of output
    # Path that will contain the grid images
    grid_images_path = os.path.join(output_dir, '/grid_images')
    # Path that will contain the illumination curves for the images without sample
    illumination_curves_ns_path = os.path.join(output_dir, '/illumination_curves_no_sample')
    # Path that will contain the illumination curves for the images with sample
    illumination_curves_sample_path = os.path.join(output_dir, '/illumination_curves_sample')
    # Path that contains the absorption images 
    absorption_path = os.path.join(output_dir, '/absorption')

    # Creating the directories
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        if args.save_aux_plots and not os.path.exists(grid_images_path):
            logger.info(f'Creating directory {grid_images_path}')
            os.makedirs(grid_images_path)
        if args.save_aux_plots and not os.path.exists(illumination_curves_ns_path):
            logger.info(f'Creating directory {illumination_curves_ns_path}')
            os.makedirs(illumination_curves_ns_path)
        if args.save_aux_plots and not os.path.exists(illumination_curves_sample_path):
            logger.info(f'Creating directory {illumination_curves_sample_path}')
            os.makedirs(illumination_curves_sample_path)
        if not os.path.exists(absorption_path):
            logger.info(f'Creating directory {absorption_path}')
            os.makedirs(absorption_path)
        
    # Generating grid images
    logger.info(f'Reading flat-field images from {ff_dir} and generating grid images')
    images = []
    # -------------------------------------- FLAT-FIELD IMAGES -------------------------------------------
    # Reading flat field images
    names = [os.path.join(ff_dir, f) for f in os.listdir(ff_dir) if f.endswith('.txt')]
    for name in names:
        # Applying necessary rotations to images
        img = np.rot90(np.genfromtxt(os.path.join(ff_dir, name)),1)
        images.append(img)
        if args.save_aux_plots:
            plt.imshow(img, cmap="bone")
            grid_filename = os.path.join(grid_images_path, f'grid_{name[3:-4]}.pdf')
            logger.info(f'Saving grid image for {name} in {grid_filename}')
            plt.savefig(grid_filename)
    images = np.array(images)

    # Correcting dead pixels for Flat field images images
    if args.dead_pixel_method == MEAN:
        logger.info('Correcting dead pixels to the mean value of the image for Flat Field images')
        correct_dead_pixels(images)
    else:
        logger.info('Correcting dead pixels using convolutions the image for Flat Field images')
        for img in images:
            try:
                convolution(img, selective=args.selective)
            except:
                raise Exception('If convolutions method is provided, --selective flag should be provided.')
    
    # Generating the illumination curves for all pixels in all Flat Field images.
    ic_nosample_file = open(os.path.join(output_dir, "illumination_curve_data_no_sample.txt"), "w")
    ic_nosample_params_file = open(os.path.join(output_dir,"illumination_curve_params_no_sample.txt"), "w")
    if args.method == FOURIER_METHOD:
        ic_nosample_params_file.write('i,j,amplitude,phase_shift,mean_value')
    else:
        # Refer to sin(x,a,b,c,d) function documentation to view the value of each parameter
        ic_nosample_params_file.write('i,j,a,b,c,d')
    logger.info('Generating the illumination curve data for all pixels in Flat Field Images')
    logger.info('Adjusting data to {args.method} method')
    illum_curves_nosample = []
    datos_fits_ns = []
    for i in range(256):
        for j in range(256):
            # Getting the profile of each pixel (i.e. intensity)
            points = profile_pixel(images,i,j)
            illum_curves_nosample.append(points)
            # Writing the intensity for all images in the pixel i,j
            ic_nosample_file.write(','.join([str(i),str(j),str(len(points)),','.join(str(p) for p in points)]))
            if args.method == FOURIER_METHOD:
                # Calculate the amplitude, phase_shift, and mean_value from fit function
                amplitude, phase_shift, mean_value = fit(illum_curves_nosample,i,j)
                ic_nosample_params_file.write(','.join([str(i), str(j), str(amplitude), str(phase_shift), str(mean_value)])) 
            else:
                # Calculate the constants for the sine fit function
                a, b, c, d = fit(illum_curves_nosample,i,j)
                ic_nosample_params_file.write(','.join([str(i), str(j), str(a), str(b), str(c), str(d)])) 
            datos_fits_ns.append(fit(illum_curves_nosample,i,j))
            # Saving the edge illumination curves to a directory
            if args.save_aux_plots:
                xdata = np.linspace(0, 2*np.pi, 100)
                plt.scatter(illum_curves_nosample[i,j], c='lightpink')
                if args.method == FOURIER_METHOD:
                    plt.plot(xdata* len(images) / (2 * np.pi), func_fourier(xdata, amplitude, phase_shift, mean_value))
                else:
                    plt.plot(xdata, sin(xdata, a, b, c, d))
                plt.xlabel('# of image')
                plt.ylabel('Intensity')
                ic_ns_filename = os.path.join(illumination_curves_ns_path, f'illumination_curve_no_sample_{i}_{j}.pdf')
                logger.info(f'Saving illumination curve for pixel {i},{j} in {ic_ns_filename}')
            if i%10 == 0 and j%10 == 0: logger.info(f'Finished processing image {i}, {j}')
    ic_nosample_file.close()
    ic_nosample_params_file.close()

    illum_curves_nosample = np.array(illum_curves_nosample)
    illum_curves_nosample = illum_curves_nosample.reshape((256,256,len(names)))
    datos_fits_ns = np.array(datos_fits_ns)
    datos_fits_ns = datos_fits_ns.reshape((256,256,3 if args.method == FOURIER_METHOD else 4))

    # ---------------------------------------- RAW IMAGES ----------------------------------------------
    # 1. ABSORPTION
    logger.info(f'Reading raw images from {raw_dir}')
    images_sample = []
    names = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.txt')]
    for name in names:
        img = np.rot90(np.genfromtxt(os.path.join(raw_dir, name)), 3)
        images_sample.append(img)
    images_sample = np.array(images_sample)

    # Correcting dead pixels for RAW images
    if args.dead_pixel_method == MEAN:
        logger.info('Correcting dead pixels to the mean value of the image for raw images')
        correct_dead_pixels(images_sample)
    else:
        logger.info('Correcting dead pixels using convolutions the image for raw images')
        for img in images_sample:
            try:
                convolution(img, selective=args.selective)
            except:
                raise Exception('If convolutions method is provided, --selective flag should be provided.')

    # Saving absorption images with dead pixel correction applied
    for img in images_sample:
        plt.imshow(img, cmap="bone")
        abs_filename = os.path.join(absorption_path, f'absorption_{name[4:-4]}.pdf')
        logger.info(f'Saving absorption image for {name} in {abs_filename}')
        plt.savefig(abs_filename)
    # Generating the illumination curves for all pixels in all Raw images.
    ic_sample_file = open(os.path.join(output_dir, "illumination_curve_data_sample.txt"), "w")
    ic_sample_params_file = open(os.path.join(output_dir,"illumination_curve_params_sample.txt"), "w")
    if args.method == FOURIER_METHOD:
        ic_sample_params_file.write('i,j,amplitude,phase_shift,mean_value')
    else:
        # Refer to sin(x,a,b,c,d) function documentation to view the value of each parameter
        ic_sample_params_file.write('i,j,a,b,c,d')
    logger.info('Generating the illumination curve data for all pixels in Raw Images')
    logger.info('Adjusting data to {args.method} method')
    illum_curves_sample = []
    datos_fits_s = []
    for i in range(256):
        for j in range(256):
            # Getting the profile of each pixel (i.e. intensity)
            points = profile_pixel(images_sample,i,j)
            illum_curves_sample.append(points)
            # Writing the intensity for all images in the pixel i,j
            ic_sample_file.write(','.join([str(i),str(j),str(len(points)),','.join(str(p) for p in points)]))
            if args.method == FOURIER_METHOD:
                # Calculate the amplitude, phase_shift, and mean_value from fit function
                amplitude, phase_shift, mean_value = fit(illum_curves_sample,i,j)
                ic_sample_params_file.write(','.join([str(i), str(j), str(amplitude), str(phase_shift), str(mean_value)])) 
            else:
                # Calculate the constants for the sine fit function
                a, b, c, d = fit(illum_curves_sample,i,j)
                ic_sample_params_file.write(','.join([str(i), str(j), str(a), str(b), str(c), str(d)])) 
            datos_fits_s.append(fit(illum_curves_sample,i,j))
            # Saving the edge illumination curves to a directory
            if args.save_aux_plots:
                xdata = np.linspace(0, 2*np.pi, 100)
                plt.scatter(illum_curves_sample[i,j], c='lightpink')
                if args.method == FOURIER_METHOD:
                    plt.plot(xdata* len(images) / (2 * np.pi), func_fourier(xdata, amplitude, phase_shift, mean_value))
                else:
                    plt.plot(xdata, sin(xdata, a, b, c, d))
                plt.xlabel('# of image')
                plt.ylabel('Intensity')
                ic_s_filename = os.path.join(illumination_curves_sample_path, f'illumination_curve_sample_{i}_{j}.pdf')
                logger.info(f'Saving illumination curve for pixel {i},{j} in {ic_s_filename}')
            if i%10 == 0 and j%10 == 0: logger.info(f'Finished processing pixel {i}, {j}')
    ic_sample_file.close()
    ic_sample_params_file.close()

    illum_curves_sample = np.array(illum_curves_sample)
    illum_curves_sample = illum_curves_sample.reshape((256,256,len(names)))
    datos_fits_s = np.array(datos_fits_s)
    datos_fits_s = datos_fits_s.reshape((256,256,3 if args.method == FOURIER_METHOD else 4))

    # 
