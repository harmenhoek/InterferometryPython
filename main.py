import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import time
from configparser import ConfigParser
import logging
import json
import git

from surface_method import method_surface, setfouriersettings
from line_method import method_line
from general_functions import TimeRemaining, image_resize_percentage, image_resize, list_images, \
    get_timestamps, timestamps_to_deltat, check_outputfolder, image_preprocessing, verify_settings, \
    conversion_factors


def main():
    __author__ = 'Harmen Hoek'
    __version__ = '0.2'

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')  # CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET

    logging.info('Code started')
    logging.info(f'Author: {__author__}, version: {__version__}.')
    start_main = time.time()
    stats = {}  # save statistics of this analysis

    # read settings from config file
    config = ConfigParser()
    config.read("config.ini")

    # create unique id (proc) for this analysis and create the folder structure if it does not exist yet
    Folders, Proc = check_outputfolder(config)
    logging.info(f'Save folder created: {Folders["savepath"]}. Process id: {Proc}.')

    # list all supported images (tiff, png, jpg, jpeg, bmp) if source is folder. If source is file, check if supported.
    inputImages, inputFolder, inputImagesFullPath = list_images(config.get("GENERAL", "SOURCE"))
    Folders['input'] = inputFolder

    # get timestamps of all frames and time difference between frames.
    timestamps, deltatime = get_timestamps(config, inputImages, inputImagesFullPath)

    # the surface_method has a simple-mode that if in use needs to set the advanced fourier settings in the config file
    if config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'surface':
        setfouriersettings(config)

    # determine conversion factors and unit-strings based on LENS_PRESET, WAVELENGTH, REFRACTIVE_INDEX and set units
    conversionFactorXY, conversionFactorZ, unitXY, unitZ = conversion_factors(config)

    stats['About'] = {}
    stats['About']['__author__'] = __author__
    stats['About']['__version__'] = __version__
    stats['About']['repo'] = str(git.Repo(search_parent_directories=True))
    stats['About']['sha'] = git.Repo(search_parent_directories=True).head.object.hexsha
    stats['startDateTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f %z')
    stats['Proc'] = Proc
    stats['inputImages'] = inputImages
    stats['inputImagesFullPath'] = inputImagesFullPath
    stats['timestamps'] = None if timestamps is None else \
        [datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in timestamps]
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
        logging.info(f"{idx + 1}/{len(inputImages)} - Analyzing started.")
        stats['analysis'][idx] = {}

        if config.get("GENERAL", "ANALYSIS_RANGE") == 'False' or idx % config.getint("GENERAL", "ANALYSIS_RANGE") == 0:

            imagePath = os.path.join(inputFolder, inputImage)
            im_gray, im_raw = image_preprocessing(config, imagePath)
            logging.info(f"{idx + 1}/{len(inputImages)} - Pre-processing done.")

            savename = f'{os.path.splitext(os.path.basename(imagePath))[0]}_analyzed_'
            stats['analysis'][idx]['imagePath'] = imagePath
            stats['analysis'][idx]['imageName'] = os.path.basename(imagePath)
            stats['analysis'][idx]['savename'] = savename

            if config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'surface':
                unwrapped_object = method_surface(config, im_gray=im_gray, conversionFactorXY=conversionFactorXY,
                                                  conversionFactorZ=conversionFactorZ, unitXY=unitXY, unitZ=unitZ,
                                                  Folders=Folders, savename=savename)
            elif config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'line':
                unwrapped_object = method_line(config, im_gray=im_gray, im_raw=im_raw,
                                               conversionFactorXY=conversionFactorXY,
                                               conversionFactorZ=conversionFactorZ, unitXY=unitXY, unitZ=unitZ,
                                               Folders=Folders, savename=savename)

            # Save unwrapped image = main result
            wrappedPath = False
            if config.getboolean("SAVING", "SAVE_UNWRAPPED_RAW_NPY"):
                wrappedPath = os.path.join(Folders['npy'], f"{savename}_unwrapped.npy")
                with open(os.path.join(Folders['npy'], f"{savename}_unwrapped.npy"), 'wb') as f:
                    np.save(f, unwrapped_object)
                stats['analysis'][idx]['wrappedPath_npy'] = os.path.relpath(wrappedPath, (Folders['save']))  # only return the relative path to main save folder
                logging.info(f'Saved unwrapped image to file with filename {wrappedPath}')
            if config.getboolean("SAVING", "SAVE_UNWRAPPED_RAW_CSV") and not config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'surface':
                wrappedPath = os.path.join(Folders['csv'], f"{savename}_unwrapped.csv")
                unwrapped_object.tofile(wrappedPath, sep=',')
                stats['analysis'][idx]['wrappedPath_csv'] = os.path.relpath(wrappedPath, (Folders['save']))  # only return the relative path to main save folder

            if config.getboolean("PLOTTING", "SHOW_PLOTS"):
                plt.show()

            plt.close('all')
        else:
            logging.info("Image skipped by user setting.")

        stats['analysis'][idx]['timeElapsed'] = time.time() - start

        timetracker.append(time.time() - start)  # add elapsed time to timetracker array
        TimeRemaining(timetracker,
                      len(inputImages) - idx)  # estimate remaining time based on average time per iteration and iterations left
        logging.info(f"{idx + 1}/{len(inputImages)} - Finished analyzing image.")

    stats['Folders'] = Folders
    stats['analysisTimeElapsed'] = time.time() - start_main
    # Save statistics
    with open(os.path.join(Folders['save'], f"{Proc}_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    # copy config file and add to export folder.
    if config.getboolean("SAVING", "SAVE_SETTINGS_TXT"):
        stats['configPath'] = os.path.join(Folders['save'], f'config_{Proc}.ini')
        with open(os.path.join(Folders['save'], f'config_{Proc}.ini'), 'w') as configfile:
            config.write(configfile)

    logging.info(f"Code finished in {round(time.time() - start_main)} seconds.")


if __name__ == "__main__":
    main()
