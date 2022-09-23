import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from natsort import natsorted
import re
import cv2
from fringe_analysis import add_crossline, filterFourierManually
from skimage.restoration import unwrap_phase
import os
import statistics
import time
from configparser import ConfigParser
import logging
from plotting import plot_process, plot_surface, plot_imwrapped, plot_imunwrapped
import json
import git

__author__ = 'Harmen Hoek'
__version__ = '0.1'

def TimeRemaining(arraytimes, left):
    avgtime = statistics.mean(arraytimes)
    timeremaining = left * avgtime
    if timeremaining < 2:
        rem = f"Almost done now ..."
    elif timeremaining < 90:
        rem = f"{round(timeremaining)} seconds"
    elif timeremaining < 3600:
        rem = f"{round(timeremaining / 60)} minutes"
    else:
        rem = f"{round(timeremaining / 3600)} hours"
    print(f"{datetime.now().strftime('%H:%M:%S')} Estimated time remaining: {rem}")
    return True

def image_resize_percentage(image, scale_percent):
    new_width = int(image.shape[1] * scale_percent / 100)
    new_height = int(image.shape[0] * scale_percent / 100)
    new_dim = (new_width, new_height)
    logging.debug(f"Image ({image.shape[1]} x {image.shape[0]}) resized to ({new_width} x {new_height}.)")
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    logging.debug(f"Image ({w} x {h}) resized to ({dim[0]} x {dim[1]}.)")
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized

def list_images(source):
    if os.path.isdir(source):
        images = os.listdir(source)
        folder = source
    else:
        images = [os.path.basename(source)]
        folder = os.path.dirname(source)
    images = [img for img in images if
              img.endswith(".tiff") or img.endswith(".png") or img.endswith(".jpg") or img.endswith(
                  ".jpeg") or img.endswith(".bmp")]
    if not images:
        raise Exception(
            f"{datetime.now().strftime('%H:%M:%S')} No images with extension '.tiff', '.png', '.jpg', '.jpeg' or '.bmp' found in selected folder.")
    return natsorted(images), folder, [os.path.join(folder, i) for i in natsorted(images)]

def timestamps_from_filenames(filenames):
    '''
    Checks for 14-digit number in filename with timestamp format %d%m%Y%H%M%S.
    Throws Exception if not found.
    TODO make it work with other fileformats.
    '''
    t = []
    for f in filenames:
        match = re.search(r"[0-9]{14}", f)  # we are looking for 14 digit number
        if not match:
            raise Exception(f"{datetime.now().strftime('%H:%M:%S')} No 14-digit timestamp found in filename.")
        try:
            t.append(datetime.strptime(match.group(0), '%m%d%Y%H%M%S'))
        except:
            raise Exception(
                f"{datetime.now().strftime('%H:%M:%S')} Could not obtain a %d%m%Y%H%M%S timestamp from the 14-digit number in filename.")
    return t

def timestamps_to_deltat(timestamps):
    deltat = [0]
    for idx in range(1, len(timestamps)):
        deltat.append((timestamps[idx] - timestamps[idx - 1]).total_seconds())
    return deltat

def check_outputfolder(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    proc = f"PROC_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if config.getboolean("SAVING", "SAVEFOLDER_CREATESUB"):
        foldername = os.path.join(foldername, proc)
        os.mkdir(foldername)

    return os.path.abspath(foldername), proc

def image_preprocessing(imagepath):
    im_raw = cv2.imread(imagepath)
    if config.get("IMAGE_PROCESSING", "IMAGE_ROTATE") != 'False':
        import imutils
        im_raw = imutils.rotate(im_raw, angle=config.getint("IMAGE_PROCESSING", "IMAGE_ROTATE"))
        logging.debug('Image rotated.')
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_RESIZE"):
        im_raw = image_resize_percentage(im_raw, 50)
        logging.debug('Image resized.')
    im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    if config.getboolean("IMAGE_PROCESSING", "CROP_DEADEDGE"):
        # im_gray = im_gray[:, 32:]
        im_gray = im_gray[8:, 28:]
        logging.debug('Image deadedge removed.')
    if config.getboolean("IMAGE_PROCESSING", "CONTRAST_ENHANCE"):
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE()
        im_gray = clahe.apply(im_gray)
        logging.debug('Image contrast enhanced.')
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_DENOISE"):
        cv2.fastNlMeansDenoising(im_gray)
        logging.debug('Image denoised.')
    return im_gray, im_raw

def setfouriersettings(config):
    '''
    If config.FOURIER_ADVANCED.ADVANCED_MODE is on, nothing happens and the ADVANCED_MODE settings are used.
    If ADVANCED_MODE is off, then the 'preset' values (lowpass and highpass filter) settings from FOURIER are used. In
    that case the FOURIER_ADVANCED settings are overwritten.
    '''
    if not config.getboolean("FOURIER_ADVANCED", "ADVANCED_MODE"):
        if config.getboolean("FOURIER", "HIGHPASS_FILTER"):
            config["FOURIER_ADVANCED"]["ROI_EDGE"] = config.getint("FOURIER", "HIGHPASS_CUTOFF")
            config["FOURIER_ADVANCED"]["BLUR"] = config.getint("FOURIER", "HIGHPASS_BLUR")
            config["FOURIER_ADVANCED"]["KEEP_SELECTION"] = True
            config["FOURIER_ADVANCED"]["SHIFTFFT"] = False
            config["FOURIER_ADVANCED"]["ROI_SECTION"] = 'lefthalf'
            config["FOURIER_ADVANCED"]["MASK_TYPE"] = 'rectangle'
        if config.getboolean("FOURIER", "LOWPASS_FILTER"):
            config["FOURIER_ADVANCED"]["SECOND_FILTER"] = True
            config["FOURIER_ADVANCED"]["ROI_EDGE_2"] = config.getint("FOURIER", "HIGHPASS_CUTOFF")
            config["FOURIER_ADVANCED"]["BLUR_2"] = config.getint("FOURIER", "HIGHPASS_BLUR")
            config["FOURIER_ADVANCED"]["KEEP_SELECTION_2"] = False
            config["FOURIER_ADVANCED"]["SHIFTFFT_2"] = False
            config["FOURIER_ADVANCED"]["ROI_SECTION_2"] = 'lefthalf'
            config["FOURIER_ADVANCED"]["MASK_TYPE_2"] = 'ellipse'
        else:
            config["FOURIER_ADVANCED"]["SECOND_FILTER"] = False
    else:
        logging.warning('Advanced fourier mode is active, highpass and lowpass filter settings are NOT used.')


def verify_settings(config, stats):
    # plotting on, while more than 1 image
    if len(stats['inputImages']) > 1 and config.getboolean("PLOTTING", "SHOW_PLOTS"):
        logging.warning(f"There are {len(stats['inputImages'])} images to be analyzed and SHOW_PLOTS is True.")
        # TODO prompt with timeout?

    # wavelength should be in nm
    if config.getfloat('GENERAL', 'WAVELENGTH') < 1:
        logging.error('WAVELENGTH should be set in nm, not meters (number is too small).')
        return False

    return True


def conversion_factors(config):
    units = ['nm', 'um', 'mm', 'm']
    conversionsXY = [1e6, 1e3, 1, 1e-3]  # standard unit is um
    conversionsZ = [1, 1e-3, 1e-6, 1e-9]  # standard unit is nm

    # Determine XY conversion factor and unit
    try:
        conversionFactorXY = config.getfloat('LENS_PRESETS', config.get('GENERAL', 'LENS_PRESET'))
        logging.info(f"Lens preset '{config.getfloat('LENS_PRESETS', config.get('GENERAL', 'LENS_PRESET'))}' is used.")
    except ValueError:
        logging.error(f"The set lens preset '{config.get('GENERAL', 'LENS_PRESET')}' is not in LENS_PRESETS.")

    unitXY = config.get('GENERAL', 'UNIT_XY')
    if unitXY not in units:
        raise ValueError(f"Desired unit {unitXY} is not valid. Choose, nm, um, mm or m.")
    conversionFactorXY = 1 / conversionFactorXY * conversionsXY[units.index(unitXY)]  # apply unit conversion

    # Determine Z conversion factor and unit
    unitZ = config.get('GENERAL', 'UNIT_Z')
    if unitZ == 'pi':
        conversionFactorZ = 1
    else:
        conversionFactorZ = (config.getfloat('GENERAL', 'WAVELENGTH')) / (4 * config.getfloat('GENERAL', 'REFRACTIVE_INDEX'))  # 1 pi = lambda / (4n). this is conversion factor in pi --> m
        if unitZ not in units:
            raise ValueError(f"Desired unit {unitZ} is not valid. Choose, nm, um, mm or m.")
        conversionFactorZ = conversionFactorZ * conversionsZ[units.index(unitZ)]  # apply unit conversion

    return conversionFactorXY, conversionFactorZ, unitXY, unitZ


logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s', datefmt='%H:%M:%S')  # CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
logging.info('Code started')
logging.info(f'Author: {__author__}, version: {__version__}.')
start_main = time.time()
stats = {}  # save statistics of this analysis

config = ConfigParser()
config.read("config.ini")


SaveFolder, Proc = check_outputfolder(config.get("SAVING", "SAVEFOLDER"))
logging.info(f'Save folder created: {SaveFolder}. Process id: {Proc}.')
if config.getboolean("GENERAL", "SAVE_SETTINGS_TXT"):
    stats['configPath'] = os.path.join(SaveFolder, f'config_{Proc}.ini' )
    with open(os.path.join(SaveFolder, f'config_{Proc}.ini'), 'w') as configfile:
        config.write(configfile)

inputImages, inputFolder, inputImagesFullPath = list_images(config.get("GENERAL", "SOURCE"))
if config.getboolean("GENERAL", "TIMESTAMPS_FROMFILENAME"):
    timestamps = timestamps_from_filenames(inputImages)
    deltatime = timestamps_to_deltat(timestamps)
    logging.info("Timestamps read from filenames. Deltatime calculated based on timestamps.")
else:
    timestamps = None
    deltatime = np.arange(0, len(inputImages)) * config.getfloat("GENERAL", "INPUT_FPS")
    logging.info("Deltatime calculated based on fps.")

setfouriersettings(config)
conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)

stats['About'] = {}
stats['About']['__author__'] = __author__
stats['About']['__version__'] = __version__
stats['About']['repo'] = str(git.Repo(search_parent_directories=True))
stats['About']['sha'] = git.Repo(search_parent_directories=True).head.object.hexsha
stats['startDateTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f %z')
stats['SaveFolder'] = SaveFolder
stats['Proc'] = Proc
stats['inputFolder'] = inputFolder
stats['inputImages'] = inputImages
stats['inputImagesFullPath'] = inputImagesFullPath
stats['timestamps'] = timestamps
stats['deltatime'] = list(deltatime)
stats['conversionFactorXY'] = conversionFactorXY
stats['conversionFactorZ'] = conversionFactorZ
stats['unitXY'] = unitXY
stats['unitZ'] = unitZ
stats['analysis'] = {}

if not verify_settings(config, stats):
    exit()

timetracker = []
for idx, inputImage in enumerate(inputImages):
    start = time.time()  # start timer to calculate iteration time
    logging.info(f"{idx+1}/{len(inputImages)} - Analyzing started.")

    imagePath = os.path.join(inputFolder, inputImage)
    im_gray, im_raw = image_preprocessing(imagePath)
    logging.info(f"{idx + 1}/{len(inputImages)} - Pre-processing done.")

    im_filtered, im_fft, im_fft_filtered, roi = \
                filterFourierManually(im_gray,
                                      keep_selection=config.getboolean("FOURIER_ADVANCED", "KEEP_SELECTION"),
                                      shiftfft=config.getboolean("FOURIER_ADVANCED", "SHIFTFFT"),
                                      blur=config.getint("FOURIER_ADVANCED", "BLUR"),
                                      roi_edge=config.getint("FOURIER_ADVANCED", "ROI_EDGE"),
                                      roi_section=config.get("FOURIER_ADVANCED", "ROI_SECTION"),
                                      mask_type=config.get("FOURIER_ADVANCED", "MASK_TYPE"),
                                      )
    logging.info(f"{idx + 1}/{len(inputImages)} - Fourier filtering 1 done.")

    if config.getboolean("FOURIER_ADVANCED", "SECOND_FILTER"):
        im_filtered, im_fft, im_fft_filtered, roi = \
            filterFourierManually(im_filtered,
                                  keep_selection=config.getboolean("FOURIER_ADVANCED", "KEEP_SELECTION_2"),
                                  shiftfft=config.getboolean("FOURIER_ADVANCED", "SHIFTFFT_2"),
                                  blur=config.getint("FOURIER_ADVANCED", "BLUR_2"),
                                  roi_edge=config.getint("FOURIER_ADVANCED", "ROI_EDGE_2"),
                                  roi_section=config.get("FOURIER_ADVANCED", "ROI_SECTION_2"),
                                  mask_type=config.get("FOURIER_ADVANCED", "MASK_TYPE_2"),
                                  )
        logging.info(f"{idx + 1}/{len(inputImages)} - Fourier filtering 2 done.")


    im_wrapped = np.arctan2(im_filtered.imag, im_filtered.real)
    im_unwrapped = unwrap_phase(im_wrapped)


    logging.info(f"{idx + 1}/{len(inputImages)} - Image wrapped and unwrapped.")

    if config.getboolean("PLOTTING", "FLIP_SURFACE"):
        im_unwrapped = -im_unwrapped + np.max(im_unwrapped)
        logging.debug('Image surface flipped.')

    im_unwrapped = im_unwrapped * conversionFactorZ
    logging.debug('Conversion factor for Z applied.')

    # Plotting
    fig1 = plot_process(im_fft, im_fft_filtered, im_gray, im_filtered, im_wrapped, im_unwrapped, roi)
    fig2 = plot_surface(im_unwrapped, config, conversionFactorXY, unitXY, unitZ)
    fig3 = plot_imwrapped(im_wrapped, config, conversionFactorXY, unitXY)
    fig4 = plot_imunwrapped(im_unwrapped, config, conversionFactorXY, unitXY, unitZ)
    logging.info(f"{idx + 1}/{len(inputImages)} - Plotting done.")


    # Saving
    savename = f'{os.path.splitext(os.path.basename(imagePath))[0]}_analyzed_'

    if config.getboolean("SAVING", "SAVE_PNG"):
        fig1.savefig(os.path.join(SaveFolder, f"process_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"unwrapped3d_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"wrapped_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        logging.debug('PNG saving done.')
    if config.getboolean("SAVING", "SAVE_PDF"):
        fig1.savefig(os.path.join(SaveFolder, f"process_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"unwrapped3d_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"wrapped_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        logging.debug('PDF saving done.')
    logging.info(f"{idx + 1}/{len(inputImages)} - Plotting and saving done.")

    # Save unwrapped image = main result
    wrappedPath = False
    if config.getboolean("SAVING", "SAVE_UNWRAPPED_RAW"):
        wrappedPath = os.path.join(SaveFolder, f"{savename}_unwrapped.npy")
        with open(os.path.join(SaveFolder, f"{savename}_unwrapped.npy"), 'wb') as f:
            np.save(f, im_unwrapped)
        logging.info(f'Saved unwrapped image to file with filename {wrappedPath}')

    stats['analysis'][idx] = {}
    stats['analysis'][idx]['imagePath'] = imagePath
    stats['analysis'][idx]['savename'] = savename
    stats['analysis'][idx]['wrappedPath'] = wrappedPath
    stats['analysis'][idx]['timeElapsed'] = time.time() - start

    if config.getboolean("PLOTTING", "SHOW_PLOTS"):
        plt.show()

    plt.close('all')

    timetracker.append(time.time() - start)  # add elapsed time to timetracker array
    TimeRemaining(timetracker, len(inputImages) - idx)  # estimate remaining time based on average time per iteration and iterations left
    logging.info(f"{idx + 1}/{len(inputImages)} - Finished analyzing image.")

stats['analysisTimeElapsed'] = time.time() - start_main
# Save statistics
with open(os.path.join(SaveFolder, f"{Proc}_statistics.json"), 'w') as f:
    json.dump(stats, f, indent=4)

logging.info(f"Code finished in {round(time.time() - start_main)} seconds.")