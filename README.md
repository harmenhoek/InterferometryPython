# InterferometryPython
 
Documentation not finished.


There are 2 analysis methods:
1. Surface
2. Line

## Analysis method: surface
With this method the whole surface is analyzed using 2D Fourier transforms.
Great for: surface plots and undulating surfaces
Not so great for: fringes running in more than 2 directions and for analyzing wide fringes.

## Analysis method: line
With this method only a slice is analyzed. Between 2 images coordinate a slice is taken. Parallel to this slice it is possible to include multiple parallel slices into the average.
Filtering is done in the Fourier space and similar to the surface analysis method the wrapped space of the surface is calculated, which is then unwrapped.
Great for: fringes running in more than 2 directions, analyzing wide fringes (with higher noise levels).
Not so great for: surface plots (not possible) and undulating surface (since the 2D information is missing in unwrapping the surface).