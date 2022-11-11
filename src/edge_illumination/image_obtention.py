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
output_dir = args.output_dir

if args.method == INTENSITIES:
    print('Hola')
else:
    # Defining subdirectories of output
    # Path that will contain the grid images
    grid_images_path = os.path.join(output_dir, '/grid_images')
    # Path that will contain the illumination curves for the images without sample
    illumination_curves_ns_path = os.path.join(output_dir, '/illumination_curves_ns')

    # Creating the directories
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        if not os.path.exists(grid_images_path):
            os.makedirs(grid_images_path)
        if not os.path.exists(illumination_curves_ns_path):
            os.makedirs(illumination_curves_ns_path)
    

    # Generating grid images
    logger.info('Generating grid images')
    images = []
    names = [os.path.join(ff_dir, f) for f in os.listdir(ff_dir) if f.endswith('.txt')]
    for name in names:
        img = np.rot90(np.genfromtxt(os.path.join(ff_dir, name)),1)
        images.append(img)
        if args.save_aux_plots:
            plt.imshow(img, cmap="bone")
            grid_filename = os.path.join(grid_images_path, f'grid_{name[3:-4]}.pdf')
            logger.info(f'Saving grid image for {name} in {grid_filename}')
            plt.savefig(grid_filename)
    images = np.array(images)

    # Correcting dead pixels
    logger.info('Correcting dead pixels to the mean value of the image')
    if args.dead_pixel_method == MEAN:
        correct_dead_pixels(images)
    else:
        for img in images:
            try:
                convolution(img, selective=args.selective)
            except:
                raise Exception('If convolutions method is provided, --selective flag should be provided.')
    
    # Generating the illumination curves for all pixels in all images.
    ic_nosample_file = open("illumination_curve_data_no_sample.txt", "w")
    ic_nosample_params_file = open("illumination_curve_params_no_sample.txt", "w")
    if args.method == FOURIER_METHOD:
        ic_nosample_params_file.write('i,j,amplitude,phase shift, mean value')
    else:
        # Refer to sin(x,a,b,c,d) documentation to view the value of each parameter
        ic_nosample_params_file.write('i,j,a,b,c,d')
    logger.info('Generating the illumination curve data for all pixels')
    logger.info('Adjusting data to {args.method} method')
    illum_curves_nosample = []
    datos_fits_ns = []
    for i in range(256):
        for j in range(256):
            points = profile_pixel(images,i,j)
            illum_curves_nosample.append(points)
            ic_nosample_file.write(','.join([str(i),str(j),str(len(points)),','.join(str(p) for p in points)]))
            if args.method == FOURIER_METHOD:
                amplitude, phase_shift,mean_value = fit(illum_curves_nosample,i,j)
                ic_nosample_params_file.write(','.join([str(i), str(j), str(amplitude), str(phase_shift), str(mean_value)])) 
            else:
                a, b, c, d = fit(illum_curves_nosample,i,j)
                ic_nosample_params_file.write(','.join([str(i), str(j), str(a), str(b), str(c), str(d)])) 
            datos_fits_ns.append(fit(illum_curves_nosample,i,j))
            if args.save_aux_plots:
                xdata = np.linspace(0, 2*np.pi, 100)
                plt.scatter(illum_curves_nosample[i,j], c='lightpink')
                if args.method == FOURIER_METHOD:
                    plt.plot(xdata* len(images) / (2 * np.pi), func_fourier(xdata, *fit(illum_curves_nosample,i,j)))
                else:
                    plt.plot(xdata, sin(xdata, *fit(illum_curves_nosample,i,j)))
                plt.xlabel('# of image')
                plt.ylabel('Intensity')
                ic_ns_filename = os.path.join(illumination_curves_ns_path, f'illumination_curve_no_sample_{i}_{j}.pdf')
                logger.info(f'Saving grid image {i},{j} in {ic_ns_filename}')
            if i%10 == 0 and j%10 == 0: logger.info(f'Finished processing image {i}, {j}')
    ic_nosample_file.close()
    ic_nosample_params_file.close()

illum_curves_nosample = np.array(illum_curves_nosample)
illum_curves_nosample = illum_curves_nosample.reshape((256,256,len(names)))
datos_fits_ns = np.array(datos_fits_ns)
datos_fits_ns = datos_fits_ns.reshape((256,256,3 if args.method == FOURIER_METHOD else 4))
