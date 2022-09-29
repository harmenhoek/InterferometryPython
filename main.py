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
from scipy import spatial

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


def method_surface(config, image):
    im_filtered, im_fft, im_fft_filtered, roi = \
        filterFourierManually(image,
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

    if config.getboolean("PLOTTING", "FLIP_UNWRAPPED"):
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
    if config.getboolean("SAVING", "SAVE_PNG"):
        fig1.savefig(os.path.join(SaveFolder, f"process_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"unwrapped3d_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"wrapped_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        logging.debug('PNG saving done.')
    if config.getboolean("SAVING", "SAVE_PDF"):
        fig1.savefig(os.path.join(SaveFolder, f"process_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"unwrapped3d_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"wrapped_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        logging.debug('PDF saving done.')
    logging.info(f"{idx + 1}/{len(inputImages)} - Plotting and saving done.")

    return im_unwrapped


def coordinates_on_line(a, b, limits):
    '''
    For a given image, a slope 'a' and offset 'b', this function gives the coordinates of all points on this
    linear line, bound by limits.
    y = a * x + b

    :param a: slope of line y=ax+b
    :param b: intersection of line with y-axis y=ax+b
    :param limits: (min x, max x, min y, max y)
    :return: zipped list of (x, y) coordinates on this line
    '''

    x = np.arange(limits[0], limits[1])  # create x coordinates
    y = (a * x + b).astype(int)  # calculate corresponding y coordinates based on y=ax+b
    # filtering is only needed for y, since x is by definition in the limits
    x = np.delete(x, (np.where((y < limits[2]) | (y >= limits[3]))))  # filter x where y is out of bound
    y = np.delete(y, (np.where((y < limits[2]) | (y >= limits[3]))))  # filter y where y is out of bound
    # return a zipped list of coordinates
    return list(zip(x, y))


def align_arrays(all_coordinates, data, alignment_coordinate):
    '''
    This function aligns arrays of different lengths to a certain (variable) alignment coordinate.
    The vacant spots are filled up with np.nan.

    :param all_coordinates: dict with all the coordinates of all the lines in list (of unequal lengths)
    :param data: correponding values to all_coordinates
    :param alignment_coordinate: the spatial coordinate to which the slices align (closest point)
    :return: 2d list with aligned lists as rows

    Example input:
    data: {
        [3, 4, 5, 3, 1, 3, 5, 6]
        [4, 5, 6, 3, 5]
        [3, 5, 6, 3, 2, 3, 4]
    }
    all_coordinates: {
        [(1,2), (2,2), (3,4), ...]
        [(2,2), (2,3), (2,4), ...]
        [...]
    }
    align_coordinate: (5,5)

    For each item in all_coordinates it will find the closest datapoint to align_coordinate. The location of
    this coordinate in all_coordinates may vary for each item. The corresponding data values for these
    locations are aligned in a 2D array. The holes (before or after or none at all) are filled up with nans.

    Example result: [
        [3, 4, 5, 3, 1, 3, 5, 6]
        [nan, 4, 5, 6, 3, 5, nan, nan]
        [3, 5, 6, 3, 2, 3, 4, nan]
    ]
    '''
    # get length of each data item list
    lengths = np.array([len(v) for _, v in data.items()])
    # for each item, calculate the index that corresponds to the smallest distance with alignment_coordinate
    alignments = np.array([spatial.KDTree(coordinates).query(alignment_coordinate)[1] for
                           _, coordinates in all_coordinates.items()])

    maxAlignment = max(alignments)  # number of nans for the array with the most nans in front
    maxPostfill = max(lengths - alignments)  # numer of nans for the array with the most nans in end
    # maxAlignment + maxPostfill = new item list length

    # create an empty 2D array for the aligned slices (dict like input is no longer needed)
    data_aligned = np.empty((len(all_coordinates), maxAlignment + maxPostfill))

    # one by one fill the new 2D array
    for kdx in np.arange(len(data)):
        profile = np.array(data[kdx])  # a data item
        part1 = np.full(maxAlignment - alignments[kdx], np.nan)  # number of nans needed before
        part3 = np.full(maxPostfill - (lengths[kdx] - alignments[kdx]), np.nan)  # number of nans needed after
        data_aligned[kdx, :] = np.hstack((part1, profile, part3)).ravel()  # new slice: nans + data item + nans

    # return the aligned 2D array
    return data_aligned

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

right_clicks = list()
def click_event(event, x, y, flags, params):
    '''
    Click event for the setMouseCallback cv2 function. Allows to select 2 points on the image and return it coordiantes.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        global right_clicks
        right_clicks.append([x, y])
    if len(right_clicks) == 2:
        cv2.destroyAllWindows()


def method_line(config, image):
    '''
    Analyzes a 2D slice using 2D fourier transforms.
    User can select 2 points in the image for the slice, or set POINTA and POINTB in the config.
    A linear line is fitted through these 2 points and beyond to the edges of the image. Next, several other slices
    parallel to this line are calculated, a total of 2*PARALLEL_SLICES_EXTENSION+1 in total. These are all averaged to
    obtain an average slice. The slice is transformed to the Fourier domain where low and high frequencies are filtered
    out. Back in spatial domain the atan2 is taken from the imag and real part, to obtain the wrapped space
    distribution. The stepped line is then unwrapped to obtain the final height profile.

    :param config: ConfigParser() config object with setting for the LINE_METHOD.
    :param image: an image to select the points for the slice on.
    :return: unwrapped line
    '''

    # get the points for the center linear slice
    if config.getboolean("LINE_METHOD", "SELECT_POINTS"):
        cv2.imshow('image', im_gray)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        global right_clicks
        P1 = right_clicks[0]
        P2 = right_clicks[1]
        logging.info(f"Selected coordinates: {P1=}, {P2=}.")
    else:
        # get from config file if preferred
        P1 = [int(e.strip()) for e in config.get('LINE_METHOD', 'POINTA').split(',')]
        P2 = [int(e.strip()) for e in config.get('LINE_METHOD', 'POINTB').split(',')]

    # get number of extra slices on each side of the center slice from config
    SliceWidth = config.getint("LINE_METHOD", "PARALLEL_SLICES_EXTENSION")


    # calculate the linear line coefficients (y=ax+b)
    x_coords, y_coords = zip(*[P1, P2])  # unzip coordinates to x and y
    a = (y_coords[1]-y_coords[0])/(x_coords[1]-x_coords[0])
    b = y_coords[0] - a * x_coords[0]

    profiles = {}  # empty dict for profiles. profiles are of different sizes of not perfectly hor or vert
    # empty dict for all coordinates on all the slices, like:
    # {[ [x1_1,y1_1],[x1_2,y1_2],...,[x1_n1,y1_n1] ], [x2_1,y2_1],[x2_2,y2_2],...,[x2_n2,y2_n2],
    # ..., [xN_1,yN_1],[xN_2,yN_2],...,[xN_nN,yN_nN] ]}
    # ni is the slice length of slice i, N is the total number of slices
    all_coordinates = {}
    for jdx, n in enumerate(range(-SliceWidth, SliceWidth + 1)):
        bn = b + n * np.sqrt(a ** 2 + 1)
        coordinates = coordinates_on_line(a, bn, [0, image.shape[1], 0, image.shape[0]])
        all_coordinates[jdx] = coordinates

        # transpose to account for coordinate system in plotting (reversed y)
        profiles[jdx] = [np.transpose(image)[pnt] for pnt in coordinates]

    # slices may have different lengths, and thus need to be aligned. We take the center of the image as the point to do
    # this. For each slice, we calculate the point (pixel) that is closest to this AlignmentPoint. We then make sure
    # that for all slices these alignments line up.
    AlignmentPoint = (image.shape[0] // 2, image.shape[1] // 2)  # take center of image as alignment
    profiles_aligned = align_arrays(all_coordinates, profiles, AlignmentPoint)





    profile = np.nanmean(profiles_aligned, axis=0)

    profile_fft = np.fft.fft(profile)  # transform to fourier space

    highPass = config.getint("LINE_METHOD_ADVANCED", "HIGHPASS_CUTOFF")
    lowPass = config.getint("LINE_METHOD_ADVANCED", "LOWPASS_CUTOFF")
    # highPassBlur = config.getint("LINE_METHOD_ADVANCED", "HIGHPASS_CUTOFF")
    # lowPassBlur = config.getint("LINE_METHOD_ADVANCED", "HIGHPASS_CUTOFF")

    mask = np.ones_like(profile).astype(float)
    if config.getboolean("LINE_METHOD_ADVANCED", "LOWPASS_FILTER"):
        mask[0:lowPass] = 0
        # mask = smooth_step(mask, lowPassBlur)
    if config.getboolean("LINE_METHOD_ADVANCED", "HIGHPASS_FILTER"):
        mask[-highPass:] = 0
        # mask = smooth_step(mask, highPassBlur)
    profile_fft = profile_fft * mask

    profile_filtered = np.fft.ifft(profile_fft)


    wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
    unwrapped = np.unwrap(wrapped)

    if config.getboolean("PLOTTING", "FLIP_UNWRAPPED"):
        unwrapped = -unwrapped + np.max(unwrapped)
        logging.debug('Image surface flipped.')

    unwrapped_converted = unwrapped * conversionFactorZ
    logging.debug('Conversion factor for Z applied.')

    # unwrapped = -unwrapped + np.max(unwrapped)

    from plotting import plot_lineprocess, plot_profiles, plot_sliceoverlay, plot_unwrappedslice
    fig1 = plot_profiles(config, profiles_aligned)
    fig2 = plot_lineprocess(config, profile, profile_filtered, wrapped, unwrapped)
    fig3 = plot_sliceoverlay(config, all_coordinates, im_raw)
    fig4 = plot_unwrappedslice(config, unwrapped_converted, profiles_aligned, conversionFactorXY, unitXY, unitZ)

    # Saving
    if config.getboolean("SAVING", "SAVE_PNG"):
        fig1.savefig(os.path.join(SaveFolder, f"rawslices_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"process_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"rawslicesimage_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.png"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        logging.debug('PNG saving done.')
    if config.getboolean("SAVING", "SAVE_PDF"):
        fig1.savefig(os.path.join(SaveFolder, f"rawslices_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"process_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"rawslicesimage_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.pdf"),
                     dpi=config.getint("SAVING", "SAVE_SETDPI"))
        logging.debug('PDF saving done.')

    plt.show()
    exit()



    return unwrapped


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
    logging.error('Settings are not valid, see errors above.')
    exit()

timetracker = []
for idx, inputImage in enumerate(inputImages):
    start = time.time()  # start timer to calculate iteration time
    logging.info(f"{idx+1}/{len(inputImages)} - Analyzing started.")

    imagePath = os.path.join(inputFolder, inputImage)
    im_gray, im_raw = image_preprocessing(imagePath)
    logging.info(f"{idx + 1}/{len(inputImages)} - Pre-processing done.")

    savename = f'{os.path.splitext(os.path.basename(imagePath))[0]}_analyzed_'
    stats['analysis'][idx] = {}
    stats['analysis'][idx]['imagePath'] = imagePath
    stats['analysis'][idx]['savename'] = savename

    if config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'surface':
        unwrapped_object = method_surface(config, im_gray)
    elif config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'line':
        unwrapped_object = method_line(config, im_gray)



    # Save unwrapped image = main result
    wrappedPath = False
    if config.getboolean("SAVING", "SAVE_UNWRAPPED_RAW"):
        wrappedPath = os.path.join(SaveFolder, f"{savename}_unwrapped.npy")
        with open(os.path.join(SaveFolder, f"{savename}_unwrapped.npy"), 'wb') as f:
            np.save(f, unwrapped_object)
        logging.info(f'Saved unwrapped image to file with filename {wrappedPath}')


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