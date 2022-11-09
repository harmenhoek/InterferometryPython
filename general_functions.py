import statistics
from datetime import datetime
import logging
import cv2
import os
from natsort import natsorted
import re
import numpy as np

def TimeRemaining(arraytimes, left):
    avgtime = statistics.mean(arraytimes)
    timeremaining = left * avgtime
    if timeremaining < 2:
        rem = f"Almost done now ..."
    elif timeremaining < 90:
        rem = f"{round(timeremaining)} seconds"
    else:
        rem = f"{round(timeremaining / 60)} minutes"
    logging.info(f"Estimated time remaining: {rem}")
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
        if not os.path.exists(source):
            logging.error(f"File {source} does not exist.")
            exit()
        images = [os.path.basename(source)]
        folder = os.path.dirname(source)
    images = [img for img in images if
              img.endswith(".tiff") or img.endswith(".png") or img.endswith(".jpg") or img.endswith(
                  ".jpeg") or img.endswith(".bmp")]
    if not images:
        raise Exception(
            f"{datetime.now().strftime('%H:%M:%S')} No images with extension '.tiff', '.png', '.jpg', '.jpeg' or '.bmp' found in selected folder.")
    return natsorted(images), folder, [os.path.join(folder, i) for i in natsorted(images)]

def get_timestamps(config, filenames, filenames_fullpath):
    '''
    For a given list of filenames, it extracts the datetime stamps from the filenames and determines the time difference
    in seconds between those timestamps. If TIMESTAMPS_FROMFILENAME is False, the set INPUT_FPS will be used to
    determine the time difference between images; timestamps will be None in that case.

    It uses standard regex expressions to look for a pattern in the filename string. Then standard c datetime format
    codes to convert that pattern into a datetime object.

    :param config: ConfigParser config object with settings
        GENERAL, TIMESTAMPS_FROMFILENAME    True or False. If False INPUT_FPS is used for deltatime, timestamps=None
        GENERAL, FILE_TIMESTAMPFORMAT_RE    The regex pattern to look for in the filename
        GENERAL, FILE_TIMESTAMPFORMAT       The datetime format code in the regex pattern for conversion
        GENERAL, INPUT_FPS                  If TIMESTAMPS_FROMFILENAME is False, use fps to determine deltatime
    :param filenames: list with filenames as strings, filenames_fullpath: list with fill path filenames as strings
    :return: timestamps, deltatime
    '''


    if config.getboolean("GENERAL", "TIMESTAMPS_FROMFILENAME"):
        timestamps = []
        for f in filenames:
            # match = re.search(r"[0-9]{14}", f)  # we are looking for 14 digit number
            match = re.search(config.get("GENERAL", "FILE_TIMESTAMPFORMAT_RE"), f)  # we are looking for 14 digit number
            if not match:
                logging.error("No 14-digit timestamp found in filename.")
                exit()
            try:
                timestamps.append(datetime.strptime(match.group(0), config.get("GENERAL", "FILE_TIMESTAMPFORMAT")))
            except:
                logging.error("Could not obtain a %d%m%Y%H%M%S timestamp from the 14-digit number in filename.")
                exit()
        deltatime = timestamps_to_deltat(timestamps)
        logging.info("Timestamps read from filenames. Deltatime calculated based on timestamps.")
    elif config.getboolean("GENERAL", "TIMESTAMPS_FROMCREATIONDATE"):
        # read from creation date file property
        timestamps = []
        for f in filenames_fullpath:
            timestamps.append(datetime.fromtimestamp(os.path.getctime(f)))
        deltatime = timestamps_to_deltat(timestamps)
        logging.info("Timestamps read from filenames. Deltatime calculated based on creation time.")
    else:
        timestamps = None
        deltatime = np.arange(0, len(filenames)) * config.getfloat("GENERAL", "INPUT_FPS")
        logging.warning("Deltatime calculated based on fps.")

    return timestamps, deltatime

def timestamps_to_deltat(timestamps):
    deltat = [0]
    for idx in range(1, len(timestamps)):
        deltat.append((timestamps[idx] - timestamps[idx - 1]).total_seconds())
    return deltat

def check_outputfolder(config):
    foldername = config.get("SAVING", "SAVEFOLDER")
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    proc = f"PROC_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if config.getboolean("SAVING", "SAVEFOLDER_CREATESUB"):
        foldername = os.path.join(foldername, proc)
        os.mkdir(foldername)

    return os.path.abspath(foldername), proc

def image_preprocessing(config, imagepath):
    im_raw = cv2.imread(imagepath)
    if config.get("IMAGE_PROCESSING", "IMAGE_ROTATE") != 'False':
        import imutils
        im_raw = imutils.rotate(im_raw, angle=config.getint("IMAGE_PROCESSING", "IMAGE_ROTATE"))
        logging.debug('Image rotated.')
    if config.get("IMAGE_PROCESSING", "IMAGE_RESIZE") != 'False':
        im_raw = image_resize_percentage(im_raw, config.getint("IMAGE_PROCESSING", "IMAGE_RESIZE"))
        logging.debug('Image resized.')
    im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    if config.get("IMAGE_PROCESSING", "IMAGE_CROP") != 'False':
        crop = [int(e.strip()) for e in config.get('IMAGE_PROCESSING', 'IMAGE_CROP').split(',')]
        im_gray = im_gray[crop[0]:-crop[2], crop[3]:-crop[1]]
        logging.debug(f'Image cropped. Pixels removed from edges: {crop}.')
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_CONTRAST_ENHANCE"):
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE()
        im_gray = clahe.apply(im_gray)
        logging.debug('Image contrast enhanced.')
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_DENOISE"):
        cv2.fastNlMeansDenoising(im_gray)
        logging.debug('Image denoised.')
    return im_gray, im_raw


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
        exit()

    unitXY = config.get('GENERAL', 'UNIT_XY')
    if unitXY not in units:
        raise ValueError(f"Desired unit {unitXY} is not valid. Choose, nm, um, mm or m.")
    conversionFactorXY = 1 / conversionFactorXY * conversionsXY[units.index(unitXY)]  # apply unit conversion

    # Determine Z conversion factor and unit
    unitZ = config.get('GENERAL', 'UNIT_Z')
    if unitZ == 'pi':
        conversionFactorZ = 1
    else:
        conversionFactorZ = (config.getfloat('GENERAL', 'WAVELENGTH')) / (2 * config.getfloat('GENERAL', 'REFRACTIVE_INDEX')) / (2 * np.pi)  # 1 period of 2pi = lambda / (4n). /2pi since our wrapped space is in absolute units, not pi
        if unitZ not in units:
            raise ValueError(f"Desired unit {unitZ} is not valid. Choose, nm, um, mm or m.")
        conversionFactorZ = conversionFactorZ * conversionsZ[units.index(unitZ)]  # apply unit conversion

    return conversionFactorXY, conversionFactorZ, unitXY, unitZ
