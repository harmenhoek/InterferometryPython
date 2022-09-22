import numpy as np
from scipy import fftpack
import cv2

def add_crossline(image):
    dims = image.shape
    cv2.line(image, (dims[1] // 2, 0), (dims[1] // 2, dims[0]),  (0, 0, 255), 1)
    cv2.line(image, (0, dims[0] // 2), (dims[1], dims[0] // 2),  (0, 0, 255), 1)
    return image

def roi_noedge(image, borderwidth):
    width, height = image.shape
    # ROI = x1, y1, width, height
    # return (borderwidth, borderwidth, height - 2 * borderwidth, width - 2 * borderwidth)  # full without edge
    return (borderwidth, borderwidth, height - 2 * borderwidth, (width // 2))  # halfway vertical
    # return (borderwidth, borderwidth, height, width - 2 * borderwidth)  # halfway horizontal
    # return (borderwidth, borderwidth, (height // 2) - 2 * borderwidth, width - 2 * borderwidth)  # halfway horizontal
    #     # return (borderwidth, borderwidth, (height // 2) - 2 * borderwidth, (width // 2) - 2 * borderwidth - 100)  # quarter only
    # return (borderwidth, borderwidth, width - 2 * borderwidth, height - 2 * borderwidth)  # working OLD

def roi_topbar(image, borderwidth):
    width, height = image.shape
    return (borderwidth, borderwidth, height - 2 * borderwidth, width // 6)

def roi_leftbar(image, borderwidth):
    width, height = image.shape
    return (borderwidth, borderwidth, height // 6, width - 2 * borderwidth)

def roi_lefthalf(image, borderwidth):
    width, height = image.shape
    return (borderwidth, borderwidth, height // 2, width - 2 * borderwidth)

def roi_all(image, borderwidth):
    width, height = image.shape
    return (borderwidth, borderwidth, height // 2, width - 2 * borderwidth)

def filterFourierManually(image, keep_selection=False, shiftfft=False, blur=False, roi_edge=10, roi_section='tophalf', mask_type='rectangle'):
    im_fft = fftpack.fft2(image)
    if shiftfft:
        im_show = fftpack.fftshift(im_fft)
    else:
        im_show = im_fft

    # im_show_2 = add_crossline(np.abs(im_show).astype(np.uint8))
    # roi = cv2.selectROI("Select ROI", im_show_2)
    # roi = (54, 22, 63, 79)
    # roi = (757, 647, 464, 97)
    # roi = (3, 0, 1934, 727)
    # roi = (7, 9, 261, 109)
    mask = np.zeros_like(im_show, dtype=np.float64)
    clr = (255, 255, 255)
    roi = False

    if mask_type == 'rectangle':
        width, height = image.shape
        if roi_section == "tophalf":
            roi = (roi_edge, roi_edge, height - 2 * roi_edge, width // 2)
        elif roi_section == 'topbar':
            roi = (roi_edge, roi_edge, height - 2 * roi_edge, width // 6)
        elif roi_section == 'leftbar':
            roi = (roi_edge, roi_edge, height // 6, width - 2 * roi_edge)
        elif roi_section == 'lefthalf':
            roi = (roi_edge, roi_edge, height // 2, width - 2 * roi_edge)
        elif roi_section == 'all':
            roi = (roi_edge, roi_edge, height - 2 * roi_edge, width - 2 * roi_edge)
        elif roi_section == 'free':
            im_show_2 = add_crossline(np.abs(im_show).astype(np.uint8))
            roi = cv2.selectROI("Select ROI", im_show_2)
        else:
            Exception('No valid mask type.')

        x1, y1, width, height = roi
        mask = cv2.rectangle(mask, (x1, y1), (x1 + width, y1 + height), clr, -1)  # thickness of -1 will fill the entire shape

    elif mask_type == 'ellipse':
        radius = min(im_show.shape) // 2 - roi_edge
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        axis_length = (mask.shape[1] // 2 - roi_edge, mask.shape[0] // 2 - roi_edge)
        # cv2.circle(mask, (cx, cy), radius, clr, -1)[0]
        cv2.ellipse(mask, (cx, cy), axis_length, 0, 0, 360, clr, -1)[0] #(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

    if blur:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    if not keep_selection:
        mask = np.abs(mask - 255)

    im_show_filtered = np.multiply(mask, im_show) / 255
    # im_show_filtered = im_show

    if shiftfft:
        im_fft_filtered = fftpack.ifftshift(im_show_filtered)
    else:
        im_fft_filtered = im_show_filtered

    im_fitlered = fftpack.ifft2(im_fft_filtered)
    return im_fitlered, im_fft, im_fft_filtered, roi