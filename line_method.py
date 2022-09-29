import numpy as np
from scipy import spatial
import cv2
import logging
import os
from matplotlib import pyplot as plt


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


def method_line(config, **kwargs):
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

    im_gray = kwargs['im_gray']
    im_raw = kwargs['im_raw']
    conversionFactorXY = kwargs['conversionFactorXY']
    conversionFactorZ = kwargs['conversionFactorZ']
    unitXY = kwargs['unitXY']
    unitZ = kwargs['unitZ']
    SaveFolder = kwargs['SaveFolder']
    savename = kwargs['savename']

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
        logging.warning(f"Coordinates for slice ({P1}, {P2}) are taken from config file.")

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
        coordinates = coordinates_on_line(a, bn, [0, im_gray.shape[1], 0, im_gray.shape[0]])
        all_coordinates[jdx] = coordinates

        # transpose to account for coordinate system in plotting (reversed y)
        profiles[jdx] = [np.transpose(im_gray)[pnt] for pnt in coordinates]

    logging.info(f"All {SliceWidth*2+1} profiles are extracted from image.")
    # slices may have different lengths, and thus need to be aligned. We take the center of the image as the point to do
    # this. For each slice, we calculate the point (pixel) that is closest to this AlignmentPoint. We then make sure
    # that for all slices these alignments line up.
    AlignmentPoint = (im_gray.shape[0] // 2, im_gray.shape[1] // 2)  # take center of image as alignment
    profiles_aligned = align_arrays(all_coordinates, profiles, AlignmentPoint)

    profile = np.nanmean(profiles_aligned, axis=0)
    logging.info("Profiles are aligned and average profile is determined.")

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
    logging.info("Average profile is filtered in the Fourier space.")


    wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
    unwrapped = np.unwrap(wrapped)
    logging.info("Average slice is wrapped and unwrapped")

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
    logging.info(f"Plotting done.")

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
    logging.info(f"Saving done.")

    return unwrapped