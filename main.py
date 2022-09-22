import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from natsort import natsorted
import re
import cv2
from fringe_analysis import add_crossline, roi_topbar, roi_noedge, roi_leftbar, filterFourierManually
from skimage.restoration import unwrap_phase
import os
import statistics
import time
from configparser import ConfigParser


from plotting import plot_process, plot_surface, plot_imwrapped, plot_imunwrapped

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

def determine_conversionfactor():
    pass

def image_preprocessing(imagepath):
    im_raw = cv2.imread(imagepath)
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_RESIZE"):
        im_raw = image_resize_percentage(im_raw, 50)
    im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    if config.getboolean("IMAGE_PROCESSING", "CROP_DEADEDGE"):
        # im_gray = im_gray[:, 32:]
        print(im_gray.shape)
        im_gray = im_gray[8:, 28:]
        print(im_gray.shape)
    if config.getboolean("IMAGE_PROCESSING", "CONTRAST_ENHANCE"):
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE()
        im_gray = clahe.apply(im_gray)
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_DENOISE"):
        cv2.fastNlMeansDenoising(im_gray)
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


def verify_settings(config):
    pass

__author__ = 'Harmen Hoek'
__version__ = '0'

# SOURCE = r'C:\Users\HOEKHJ\surfdrive\Polymerbrush_Spreading\HTK_openair_20220425\HTK_openair_20220425_raw_selection'
# SOURCE = r'C:\Users\HOEKHJ\surfdrive\Polymerbrush_Spreading\HTK_openair_20220425\HTK_openair_20220425_raw_selection\HTK_openair_20220425-04252022173701-1.tiff'
# SOURCE = r'C:\Users\HOEKHJ\surfdrive\Polymerbrush_Spreading\HTK_openair_20220425\HTK_openair_20220425_raw_selection\HTK_openair_20220425-04262022145446-502.tiff'
# SOURCE = 'images/1-04222022065406-72.tiff'

config = ConfigParser()
config.read("config.ini")

SaveFolder, Proc = check_outputfolder(config.get("SAVING", "SAVEFOLDER"))
if config.getboolean("GENERAL", "SAVE_SETTINGS_TXT"):
    with open(os.path.join(SaveFolder, f'config_{Proc}.ini'), 'w') as configfile:
        config.write(configfile)


inputImages, inputFolder, inputImagesFullPath = list_images(config.get("GENERAL", "SOURCE"))
if config.getboolean("GENERAL", "TIMESTAMPS_FROMFILENAME"):
    timestamps = timestamps_from_filenames(inputImages)
    deltatime = timestamps_to_deltat(timestamps)
else:
    timestaps = None
    deltatime = np.arange(0, len(inputImages)) * config.getfloat("GENERAL", "INPUT_FPS")

setfouriersettings(config)

timetracker = []
for idx, inputImage in enumerate(inputImages):
    start = time.time()  # start timer to calculate iteration time

    imagePath = os.path.join(inputFolder, inputImage)

    im_gray, im_raw = image_preprocessing(imagePath)

    im_filtered, im_fft, im_fft_filtered, roi = \
                filterFourierManually(im_gray,
                                      keep_selection=config.getboolean("FOURIER_ADVANCED", "KEEP_SELECTION"),
                                      shiftfft=config.getboolean("FOURIER_ADVANCED", "SHIFTFFT"),
                                      blur=config.getint("FOURIER_ADVANCED", "BLUR"),
                                      roi_edge=config.getint("FOURIER_ADVANCED", "ROI_EDGE"),
                                      roi_section=config.get("FOURIER_ADVANCED", "ROI_SECTION"),
                                      mask_type=config.get("FOURIER_ADVANCED", "MASK_TYPE"),
                                      )

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

    im_wrapped = np.arctan2(im_filtered.imag, im_filtered.real)
    im_unwrapped = unwrap_phase(im_wrapped)

    if config.getboolean("PLOTTING", "FLIP_SURFACE"):
        im_unwrapped = -im_unwrapped + np.max(im_unwrapped)

    if config.getint("GENERAL", "CONVERSION_FACTOR"):
        im_unwrapped = im_unwrapped * config.getint("GENERAL", "CONVERSION_FACTOR")

    #
    # im2 = np.divide(im_filtered.imag, im_filtered.real)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.imshow(im2, cmap='gray')
    # fig.colorbar(cax, pad=0.1, label='', shrink=0.5)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.imshow(im_filtered.imag, cmap='gray')
    # ax.set_title('imag')
    # fig.colorbar(cax, pad=0.1, label='', shrink=0.5)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.imshow(im_filtered.real, cmap='gray')
    # ax.set_title('real')
    # fig.colorbar(cax, pad=0.1, label='', shrink=0.5)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.imshow(np.arctan(im_filtered.imag, im_filtered.real), cmap='gray')
    # ax.set_title('arctan1')
    # fig.colorbar(cax, pad=0.1, label='', shrink=0.5)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.imshow(im_wrapped, cmap='gray')
    # ax.set_title('arctan2')
    # fig.colorbar(cax, pad=0.1, label='', shrink=0.5)
    #
    # plt.show()
    #
    # exit()

    # Plotting
    fig1 = plot_process(im_fft, im_fft_filtered, im_gray, im_filtered, im_wrapped, im_unwrapped, roi)
    # fig2 = plot_surface(im_unwrapped, config, overlay_image=np.flipud(im_raw/255))
    fig2 = plot_surface(im_unwrapped, config)
    fig3 = plot_imwrapped(im_wrapped, config)
    fig4 = plot_imunwrapped(im_unwrapped, config)



    # Saving
    savename = f'{os.path.splitext(os.path.basename(imagePath))[0]}_{config.getint("FOURIER_ADVANCED", "BLUR")}_{config.getint("FOURIER_ADVANCED", "ROI_EDGE")}_{config.getboolean("FOURIER_ADVANCED", "SHIFTFFT")},{config.getboolean("FOURIER_ADVANCED", "KEEP_SELECTION")}'

    if config.getboolean("SAVING", "SAVE_PNG"):
        fig1.savefig(os.path.join(SaveFolder, f"process_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"unwrapped3d_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"wrapped_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.png"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
    if config.getboolean("SAVING", "SAVE_PDF"):
        fig1.savefig(os.path.join(SaveFolder, f"process_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig2.savefig(os.path.join(SaveFolder, f"unwrapped3d_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig3.savefig(os.path.join(SaveFolder, f"wrapped_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))
        fig4.savefig(os.path.join(SaveFolder, f"unwrapped_{savename}.pdf"), dpi=config.getint("SAVING", "SAVE_SETDPI"))

    if config.getboolean("PLOTTING", "SHOW_PLOTS"):
        plt.show()

    plt.close('all')

    timetracker.append(time.time() - start)  # add elapsed time to timetracker array
    TimeRemaining(timetracker, len(inputImages) - idx)  # estimate remaining time based on average time per iteration and iterations left

