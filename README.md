# InterferometryPython

A Python program for analyzing single wavelength interferometry images based on fringe detection using Fourier space frequency filtering to obtain 1D or 2D height profiles.
 
Documentation for version 0.3 (not fully tested yet) \
Documentation last updated: 2022-09-29

## Getting started

To get started, create a virtual environment:
```
python3 -m venv /path/to/new/virtual/environment
```
Launch the venv: \
Windows: `.\venv\Scripts\activate` \
Unix: `source venv/bin/activate`

Install all the packages from `requirements.txt`:
```
pip install -r requirements.txt
```
Next, change the settings in the config file `config.ini`. Then run the main code:
```
python main.py 
```

## Input & output

Input:
- One or more images of interferometry patterns. Images need to be recorded at single wavelength. Images need to be of format: .png, .jpeg, .jpg, .tiff or .bmp.
- Experiment-specific settings: `REFRACTIVE_INDEX` (for height conversion from pixels to m), `WAVELENGTH` (for height conversion), `LENS_PRESET` (one of the `LENS_PRESETS` that contain the conversion factor pix to mm for a specific lens for conversion of x,y), `INPUT_FPS` or timestamps in the images (to determine the time difference between frames).

All settings for the analysis are documented in the example settings file `config.ini`.

Output:
- Plots
- A 1D or 2D unwrapped height profile

## How it works

There are 2 analysis methods. The **line method** analyses 1D image slices to create a height profile, the **surface method** analyses the full 2d images to create a full surface map.

### Line method
A single profile slice is taken from the image based on the user settings or user GUI selection. Several parallel slices to this slice (2 x `PARALLEL_SLICES_EXTENSION`) are included in an average slice profile. Filtering is done in the Fourier space, where high and low frequencies can be removed to be left with a small band of similar frequencies. The slice is converted back to the spatial domain and the wrapped phase space is calculated as follows:
```
arctan2(FilteredImage.image, FilteredImage.real)
```
The wrapped space shows the periodic phase of the fringes as a stepped function with an amplitude of -pi to pi. Unwrapping the wrapped phase space results in the final height profile in units of pi.

Conversion of x,y is done using the conversion factor pix/mm (done with presets `LENS_PRESET` in config file).
Conversion of z of units of pi to meter is done by:
```
pi = lambda / (4n)
```
where `n` is the refractive index (`REFRACTIVE_INDEX`) and lambda the wavelength (`WAVELENGTH`).

### Surface method
The surface method works in theory similar to the line method, expect that filtering in the Fourier space needs to be done in 2 directions. This requires more fine-tuning of the settings.

### Line method vs Surface method
When to use the line method, when to use the surface method?

The line method is great for:
- 1D fringe patterns. That is, fringes that run in a single direction and are always parallel to one another.
- Height profiles that are monotonically increasing or decreasing. The line method cannot detect local extrema.
- Patterns with a wide range of frequencies. That is, small and wide fringes in a single profile. Since very little filtering is needed for 1D profiles all relevant frequencies can be preserved. The surface method requires much more filtering.

The surface method is great for:
- Creating surface plots
- Undulating surfaces. The 2d unwrapping process in the surface method detects local extrema.

Neither method works for:
- Fringes running in more than 2 directions, i.e. fringes in just x and y work, but also including diagonal fringes makes simple global filtering difficult. Ideally you'd need spatial dependent frequency filtering for these cases. Consider splitting the image into several simpler patterns to overcome this problem.

## Screenshots

Examples of the line method and surface method are shown in the images below.

![Example of the line method.](screenshots/line_method_example.jpeg?raw=true "Example of the line method.")
![Example of the surface method.](screenshots/surface_method_example.jpeg?raw=true "Example of the surface method.")

## Troubleshooting

### [surface_method] Discontinuities in surface plots
Consider splitting the interferometery image up into several smaller images.\
When fringes run in more than 2 directions, i.e. they do not only have run in x and y, but also diagonally, the filtering process becomes very tricky. Consider the example in the image below. The black arrows in indicate the frequencies we are interested in, i.e. the frequencies that run perpendicular to the fringes. However, the frequency we want to keep in the purple section, also shown up in the same direction in the blue section. This means that no matter what global filtering we do, we cannot keep only the frequencies we need. This causes discontinuities in the final height profile.

![Example limitation of global 2 directional frequency filtering.](screenshots/frequency_filtering_limitation.jpeg?raw=true "Example limitation of global 2 directional frequency filtering.")

## License
MIT License

Copyright (c) 2022 harmenhoek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Work in progress
- Slicing is not perfect. Parallel slices are sometimes 2pi apart. Alignment is terrible in some cases. Ideally we would want the option to automatically align the slices perpendicular to the fringes, and also align the slices close to the contact line.
- More data validation. Develop algorithms to interpret the data and determine the validity of it. This includes data far away from the CL. If the contrast e.g. is too low here, or the fringes are too far away, we should ignore these datapoints.

Small todos:
- Don't save absolute paths in JSON, only relative
- store data in subfolders