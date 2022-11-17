from inline_utils import *
from inline_logger import *
import matplotlib.pyplot as plt
import cv2
import os

from argparse import ArgumentParser

# Definition of the arguments for the parser
ap = ArgumentParser(description = 'Inline: image generation', 
            epilog='This scipt generates attenuation, visibility and phase contrast images for the inline method')
ap.add_argument('--root-dir',dest='root_dir', help='root directory with all the data', required=True)
ap.add_argument('--ff-dir',dest='ff_dir', default='2. FF/', help='path to data with the Flat Field data')
ap.add_argument('--raw-dir',dest='raw_dir', default='1. RAW/', help='path to data with the Raw image data')
ap.add_argument('--output-dir',dest='output_dir', default='./output/', help='path to output directory')
ap.add_argument('--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                default='DEBUG', help='Set log level (default DEBUG)')
ap.add_argument('--save-aux-plots', dest='save_aux_plots', action='store_const', const=True, default=False, 
                help='If the plots for grids and illumination curves should be stored')
ap.add_argument('--method', dest='method', choices=[PAGANIN, BELTRAN],
                required=True, help='Method to be used for the image generation')
ap.add_argument('--sample', dest='sample', choices=[WATER, AIR, BLOOD, IODOPOVIDONE], default=BLOOD,
                required=True, help='Sample with which the tubes are filled')
args = ap.parse_args()

# Setting the logger
set_args(args)
logger = get_logger(__name__)


if args.sample == IODOPOVIDONE:
    delta_muestra = deltaYodo
    beta_muestra = betaYodo
elif args.sample == WATER:
    delta_muestra = deltaAgua
    beta_muestra = betaAgua
elif args.sample == AIR:
    delta_muestra = deltaAire
    beta_muestra = betaAire
else:
    delta_muestra = deltaSangreArtificial
    beta_muestra = betaSangreArtificial

# Getting parser parameters
root_dir = args.root_dir
ff_dir = os.path.join(root_dir, args.ff_dir)
raw_dir = os.path.join(root_dir, args.raw_dir)
output_dir = args.output_dir

# Defining subdirectories of output
# Path that contains the absorption images 
absorption_path = os.path.join(output_dir, 'absorption')
phase_path = os.path.join(output_dir, 'phase')
    
# Creating the directories
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)
if not os.path.exists(absorption_path):
    logger.info(f'Creating directory {absorption_path}')
    print(os.path.join(output_dir, 'absorption'))
    os.makedirs(os.path.join(output_dir, 'absorption'))
if not os.path.exists(phase_path):
    logger.info(f'Creating directory {phase_path}')
    os.makedirs(os.path.join(output_dir, 'phase'))

name1 = os.path.join(raw_dir, 'RAW.txt')

E_J=E*1.6*(10**(-16))
h=10**(-34)
c=3*(10**(8))
k=E_J/(h*c)
delta_rel=np.abs(delta_muestra-delta_m1)
beta_rel=np.abs(beta_muestra-beta_m1)
delta_rel1=np.abs(deltaAire-delta_m1)
beta_rel1=np.abs(betaAire-beta_m1)
RAW=Raws(name1)
FF=FF(59, ff_dir)
RAWC=RAW/FF
# RAWS= np.rot90(np.genfromtxt(name3),1)
# PVC = np.rot90(np.genfromtxt(name4),1)
# Esp_T=Esp(PVC, a, delta_m1, beta_m1, E, z)
# M1_E=Esp(RAWS, a, delta_m1, beta_m1, E, z)
M1_RAW=Esp(RAWC, a, delta_m1, beta_m1, E, z)
# RAW_M1=RAWS/(np.exp(-2*k*beta_m1*M1_E))
RAW_M2=RAW/(FF*np.exp(-2*k*beta_m1*M1_RAW))
# M_RAW=Esp(RAW_M1, a, delta_rel, beta_rel, E, z)
M_RAW1=Esp(RAW_M2, a, delta_rel, beta_rel, E, z)
M_RAW2=Esp(RAWC, a, delta_muestra, beta_muestra, E, z)
# M_RAW3=Esp(RAWS, a, delta_muestra, beta_muestra, E, z)
M_RAW4=Esp(RAWC, a, delta_rel, beta_rel, E, z)
# M_RAW5=Esp(RAWS, a, delta_rel, beta_rel, E, z)
M_RAW6=Esp(RAWC, a, delta_rel1, beta_rel, E, z)
#M_RAW7=Esp(RAWS, a, delta_rel1, beta_rel, E, z)
PhaseP=-k*M_RAW2*delta_muestra
PhaseB=-k*M1_RAW* delta_m1-k*delta_rel*M_RAW1

# Saving absorption image
plt.figure(figsize=(8,8))
im = plt.imshow(abs(RAW), cmap="bone")
plt.gca().invert_yaxis()
plt.title('Imagen por atenuación para la muestra de sangre artificial del tubo de 1.43 mm\n', fontsize=14)
plt.xlabel('Número de píxel en el eje x del detector', fontsize=12)
plt.ylabel('Número de píxel en el eje y del detector', fontsize=12)
clb = plt.colorbar(im,fraction=0.046, pad=0.04)
clb.ax.get_yaxis().labelpad = 15
clb.set_label('Intensidad [a.u.]', rotation=270, fontsize=12)
plt.tight_layout()
abs_filename = os.path.join(absorption_path, f'{args.sample}_absorption.pdf')
logger.info(f'Saving absorption image for {name1} in {abs_filename}')
plt.savefig(abs_filename)

if args.method == BELTRAN:
    plt.figure(figsize=(8,8))
    im = plt.imshow(abs(PhaseB), cmap="bone")
    plt.title(f'Phase image by Beltran\'s method for the {args.sample} sample', fontsize=14)
    plt.gca().invert_yaxis()
    plt.xlabel('Pixel number in the detector\'s x axis', fontsize=12)
    plt.ylabel('Pixel number in the detector\'s y axis', fontsize=12)
    clb = plt.colorbar(im,fraction=0.046, pad=0.04)
    clb.ax.get_yaxis().labelpad = 15
    clb.set_label('Intensity [a.u.]', rotation=270, fontsize=12)
    plt.tight_layout()
    phase_filename = os.path.join(phase_path, f'{args.sample}_phase_{args.method}.pdf')
    logger.info(f'Saving phase image for {name1} in {phase_filename}')
    plt.savefig(phase_filename)

    plt.figure(figsize=(8,8))
    plt.plot(PI(PhaseB))
    plt.title("Intensity profile")
    plt.ylabel("Counts")
    plt.xlabel("Relative position")
    ip_filename = os.path.join(phase_path, f'{args.sample}_intensity_profile_{args.method}.pdf')
    logger.info(f'Saving intensity profile for {name1} in {ip_filename}')
    plt.savefig(ip_filename)


else:
    plt.figure(figsize=(8,8))
    im = plt.imshow(abs(PhaseP), cmap="bone")
    plt.title(f'Phase image by Paganin\'s method for the {args.sample} sample', fontsize=14)
    plt.gca().invert_yaxis()
    plt.xlabel('Pixel number in the detector\'s x axis', fontsize=12)
    plt.ylabel('Pixel number in the detector\'s y axis', fontsize=12)
    clb = plt.colorbar(im,fraction=0.046, pad=0.04)
    clb.ax.get_yaxis().labelpad = 15
    clb.set_label('Intensity [a.u.]', rotation=270, fontsize=12)
    plt.tight_layout()
    phase_filename = os.path.join(phase_path, f'{args.sample}_phase_{args.method}.pdf')
    logger.info(f'Saving grid image for {name1} in {phase_filename}')
    plt.savefig(phase_filename)

    plt.figure(figsize=(8,8))
    plt.plot(PI(PhaseB))
    plt.title("Intensity profile")
    plt.ylabel("Counts")
    plt.xlabel("Relative position")
    ip_filename = os.path.join(phase_path, f'{args.sample}_intensity_profile_{args.method}.pdf')
    logger.info(f'Saving intensity profile for {name1} in {ip_filename}')
    plt.savefig(ip_filename)
