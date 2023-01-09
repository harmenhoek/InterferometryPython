import numpy as np
from scipy import fftpack
import cv2
from plotting import plot_process, plot_surface, plot_imwrapped, plot_imunwrapped
import logging
from skimage.restoration import unwrap_phase
import os

def setfouriersettings(config):
    '''
    If config.SURFACE_METHOD_ADVANCED.ADVANCED_MODE is on, nothing happens and the ADVANCED_MODE settings are used.
    If ADVANCED_MODE is off, then the 'preset' values (lowpass and highpass filter) settings from FOURIER are used. In
    that case the SURFACE_METHOD_ADVANCED settings are overwritten.
    '''
    if not config.getboolean("SURFACE_METHOD_ADVANCED", "ADVANCED_MODE"):
        if config.getboolean("SURFACE_METHOD", "HIGHPASS_FILTER"):
            config["SURFACE_METHOD_ADVANCED"]["ROI_EDGE"] = config.getint("SURFACE_METHOD", "HIGHPASS_CUTOFF")
            config["SURFACE_METHOD_ADVANCED"]["BLUR"] = config.getint("SURFACE_METHOD", "HIGHPASS_BLUR")
            config["SURFACE_METHOD_ADVANCED"]["KEEP_SELECTION"] = True
            config["SURFACE_METHOD_ADVANCED"]["SHIFTFFT"] = False
            config["SURFACE_METHOD_ADVANCED"]["ROI_SECTION"] = 'lefthalf'
            config["SURFACE_METHOD_ADVANCED"]["MASK_TYPE"] = 'rectangle'
        if config.getboolean("SURFACE_METHOD", "LOWPASS_FILTER"):
            config["SURFACE_METHOD_ADVANCED"]["SECOND_FILTER"] = True
            config["SURFACE_METHOD_ADVANCED"]["ROI_EDGE_2"] = config.getint("SURFACE_METHOD", "HIGHPASS_CUTOFF")
            config["SURFACE_METHOD_ADVANCED"]["BLUR_2"] = config.getint("SURFACE_METHOD", "HIGHPASS_BLUR")
            config["SURFACE_METHOD_ADVANCED"]["KEEP_SELECTION_2"] = False
            config["SURFACE_METHOD_ADVANCED"]["SHIFTFFT_2"] = False
            config["SURFACE_METHOD_ADVANCED"]["ROI_SECTION_2"] = 'lefthalf'
            config["SURFACE_METHOD_ADVANCED"]["MASK_TYPE_2"] = 'ellipse'
        else:
            config["SURFACE_METHOD_ADVANCED"]["SECOND_FILTER"] = False
    else:
        logging.warning('Advanced fourier mode is active, highpass and lowpass filter settings are NOT used.')

def add_crossline(image):
    dims = image.shape
    cv2.line(image, (dims[1] // 2, 0), (dims[1] // 2, dims[0]),  (0, 0, 255), 1)
    cv2.line(image, (0, dims[0] // 2), (dims[1], dims[0] // 2),  (0, 0, 255), 1)
    return image

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
        height, width = image.shape
        # roi = x1, y1, width, height
        if roi_section == "tophalf":
            roi = (roi_edge, roi_edge, width - 2 * roi_edge, height // 2)
        elif roi_section == 'topbar':
            roi = (roi_edge, roi_edge, width - 2 * roi_edge, height // 6)
        elif roi_section == "bottomhalf":
            roi = (roi_edge, height // 2, width - 2 * roi_edge, height // 2 - roi_edge)
        elif roi_section == 'bottombar':
            roi = (width // 6 * 5 + roi_edge, roi_edge, width - 2 * roi_edge, height // 6)
        elif roi_section == 'leftbar':
            roi = (roi_edge, roi_edge, width // 6, height - 2 * roi_edge)
        elif roi_section == 'lefthalf':
            roi = (roi_edge, roi_edge, width // 2, height - 2 * roi_edge)
            # roi = (roi_edge, roi_edge, width // 2 + roi_edge, height - 2 * roi_edge)
            # roi = (roi_edge, roi_edge, width, height - 2 * roi_edge)
        elif roi_section == 'rightbar':
            roi = (roi_edge, height // 6 * 5 + roi_edge, width // 6, height - 2 * roi_edge)
        elif roi_section == 'righthalf':
            roi = (roi_edge, roi_edge, width // 2, height - 2 * roi_edge)
        elif roi_section == 'all':
            roi = (roi_edge, roi_edge, width - 2 * roi_edge - 1, height - 2 * roi_edge)
        elif roi_section == 'free':
            im_show_2 = add_crossline(np.abs(im_show).astype(np.uint8))
            roi = cv2.selectROI("Select ROI", im_show_2)
        elif roi_section == 'quarter1':
            roi = (roi_edge, roi_edge, width // 2, height // 2)
        elif roi_section == 'quarter2':
            roi = (roi_edge, height // 2 + roi_edge, width // 2, height // 2)
        elif roi_section == 'quarter3':
            roi = (width // 2 + roi_edge, roi_edge, width // 2, height // 2)
        elif roi_section == 'quarter4':
            roi = (width // 2 + roi_edge, height // 2 + roi_edge, width // 2, height // 2)
        elif roi_section == 'sixth1':
            pass
        elif roi_section == 'sixth2':
            pass
        elif roi_section == 'sixth3':
            pass
        elif roi_section == 'sixth4':
            pass
        elif roi_section == 'testing':
            # edge_h = 1
            # roi = (1, edge_h, width // 2 - 500, height - edge_h)

            edge_w = 2
            roi = (edge_w, 1, width - 2 * edge_w, height // 3)


        else:
            Exception('No valid mask type.')

        x1, y1, height, width = roi
        mask = cv2.rectangle(mask, (x1, y1), (x1 + height, y1 + width), clr, -1)  # thickness of -1 will fill the entire shape

    elif mask_type == 'ellipse':
        radius = min(im_show.shape) // 2 - roi_edge
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        axis_length = (mask.shape[1] // 2 - roi_edge, mask.shape[0] // 2 - roi_edge)
        # cv2.circle(mask, (cx, cy), radius, clr, -1)[0]
        cv2.ellipse(mask, (cx, cy), axis_length, 0, 0, 360, clr, -1)[0] #(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

    if blur:
        '''
        Gaussianblur(src, ksize, sigmaX, sigmaY, borderType)
        ksize	Gaussian kernel size. ksize.height and ksize.width can differ but they both must be positive and odd. 
                Or, they can be zero's and then they are computed from sigma. 
        sigmaX	Gaussian kernel standard deviation in X direction.
        sigmaY	Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, 
                if both sigmas are zeros, they are computed from ksize.height and ksize.width, respectively (see 
                getGaussianKernel for details); to fully control the result regardless of possible future modifications 
                of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
        borderType	pixel extrapolation method, see BorderTypes.
                Member name         Value   Description
                BORDER_CONSTANT     0       iiiiii|abcdefgh|iiiiiii with some specified i. Border is filled with the 
                                            fixed value, passed as last parameter of the function.
                BORDER_REPLICATE    1       aaaaaa|abcdefgh|hhhhhhh The pixels from the top and bottom rows, the 
                                            left-most and right-most columns are replicated to fill the border.
                BORDER_REFLECT      2       fedcba|abcdefgh|hgfedcb
                BORDER_REFLECT_101  4       gfedcb|abcdefgh|gfedcba
                BORDER_TRANSPARENT  5       uvwxyz|abcdefgh|ijklmno
                BORDER_DEFAULT      4       same as BORDER_REFLECT_101
                BORDER_ISOLATED     16      do not look outside of ROI      
                
        '''
        mask = cv2.GaussianBlur(mask, (blur, blur), cv2.BORDER_CONSTANT)
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

def method_surface(config, **kwargs):

    im_gray = kwargs['im_gray']
    conversionFactorXY = kwargs['conversionFactorXY']
    conversionFactorZ = kwargs['conversionFactorZ']
    unitXY = kwargs['unitXY']
    unitZ = kwargs['unitZ']
    # SaveFolder = kwargs['SaveFolder']
    Folders = kwargs['Folders']
    savename = kwargs['savename']

    im_filtered, im_fft, im_fft_filtered, roi = \
        filterFourierManually(im_gray,
                              keep_selection=config.getboolean("SURFACE_METHOD_ADVANCED", "KEEP_SELECTION"),
                              shiftfft=config.getboolean("SURFACE_METHOD_ADVANCED", "SHIFTFFT"),
                              blur=config.getint("SURFACE_METHOD_ADVANCED", "BLUR"),
                              roi_edge=config.getint("SURFACE_METHOD_ADVANCED", "ROI_EDGE"),
                              roi_section=config.get("SURFACE_METHOD_ADVANCED", "ROI_SECTION"),
                              mask_type=config.get("SURFACE_METHOD_ADVANCED", "MASK_TYPE"),
                              )
    logging.info(f"Fourier filtering 1 done.")

    if config.getboolean("SURFACE_METHOD_ADVANCED", "SECOND_FILTER"):
        im_filtered, im_fft, im_fft_filtered, roi = \
            filterFourierManually(im_filtered,
                                  keep_selection=config.getboolean("SURFACE_METHOD_ADVANCED", "KEEP_SELECTION_2"),
                                  shiftfft=config.getboolean("SURFACE_METHOD_ADVANCED", "SHIFTFFT_2"),
                                  blur=config.getint("SURFACE_METHOD_ADVANCED", "BLUR_2"),
                                  roi_edge=config.getint("SURFACE_METHOD_ADVANCED", "ROI_EDGE_2"),
                                  roi_section=config.get("SURFACE_METHOD_ADVANCED", "ROI_SECTION_2"),
                                  mask_type=config.get("SURFACE_METHOD_ADVANCED", "MASK_TYPE_2"),
                                  )
        logging.info(f"Fourier filtering 2 done.")

    im_wrapped = np.arctan2(im_filtered.imag, im_filtered.real)
    im_unwrapped = unwrap_phase(im_wrapped)

    logging.info(f"Image wrapped and unwrapped.")

    if config.getboolean("PLOTTING", "FLIP_UNWRAPPED"):
        im_unwrapped = -im_unwrapped + np.max(im_unwrapped)
        logging.debug('Image surface flipped.')

    im_unwrapped = im_unwrapped * conversionFactorZ
    logging.debug('Conversion factor for Z applied.')


    # Plotting and saving of figures
    SavePNG = config.getboolean("SAVING", "SAVE_PNG")
    SavePDF = config.getboolean("SAVING", "SAVE_PDF")

    if config.getboolean("PLOTTING", "PLOT_SURFACEMETHOD_PROCESS"):
        fig1 = plot_process(im_fft, im_fft_filtered, im_gray, im_filtered, im_wrapped, im_unwrapped, roi)
        if SavePNG:
            fig1.savefig(os.path.join(Folders['save_process'], f"process_{savename}.png"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))
        if SavePDF:
            fig1.savefig(os.path.join(Folders['save_process'], f"process_{savename}.pdf"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))

    if config.getboolean("PLOTTING", "PLOT_SURFACEMETHOD_SURFACE"):
        fig2 = plot_surface(im_unwrapped, config, conversionFactorXY, unitXY, unitZ)
        if SavePNG:
            fig2.savefig(os.path.join(Folders['save_unwrapped3d'], f"unwrapped3d_{savename}.png"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))
        if SavePDF:
            fig2.savefig(os.path.join(Folders['save_unwrapped3d'], f"unwrapped3d_{savename}.pdf"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))

    if config.getboolean("PLOTTING", "PLOT_SURFACEMETHOD_WRAPPED"):
        fig3 = plot_imwrapped(im_wrapped, config, conversionFactorXY, unitXY)
        if SavePNG:
            fig3.savefig(os.path.join(Folders['save_wrapped'], f"wrapped_{savename}.png"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))
        if SavePDF:
            fig3.savefig(os.path.join(Folders['save_wrapped'], f"wrapped_{savename}.pdf"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))

    if config.getboolean("PLOTTING", "PLOT_SURFACEMETHOD_UNWRAPPED"):
        fig4 = plot_imunwrapped(im_unwrapped, config, conversionFactorXY, unitXY, unitZ)
        if SavePNG:
            fig4.savefig(os.path.join(Folders['save_unwrapped'], f"unwrapped_{savename}.png"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))
        if SavePDF:
            fig4.savefig(os.path.join(Folders['save_unwrapped'], f"unwrapped_{savename}.pdf"),
                         dpi=config.getint("SAVING", "SAVE_SETDPI"))

    logging.info(f"Plotting (and saving) done.")



    return im_unwrapped