# OzDES_makeLC
This repository contains the code that was developed for the OzDES
Reverberation Mapping Program to make emission line light curves and
line widths.  This code accepts the photometric light curves obtained
using [OzDES_getPhoto](https://github.com/jhoormann/OzDES_getPhoto) and
the spectroscopic data calibrated using
[OzDES_calibSpec](https://github.com/jhoormann/OzDES_calibSpec).

## Functionality
The following flags, if set to true, will perform various aspects of the
analysis.
* convertPhotoLC: This will take the photometric light curves in terms
of magnitudes (ie from
 [OzDES_getPhoto](https://github.com/jhoormann/OzDES_getPhoto)) and
 convert to fluxes.  The light curves for each band will be saved in
 separate files.

 * makePhotoLC: This will apply specified photometric transmission
 functions to the spectra to create light curves for the bands.

 * makeLineLC: For the list of available emission lines the emission
 line fluxes will be calculated after local continuum subtraction.

 * calcWidth:  The width of the available emission lines will be
 calculated via the FWHM and velocity dispersion on both the mean
 and RMS spectrum.

 * calcBH: Given the line width measurements from the above flag the
 black hole masses are calculated using the Radius-Luminosity
 relationships.

 * makeFig/makeFigEpochs:  Makes and saves diagnostic plots.

# Run Requirements
The code was tested using the following (as stated in requirements.txt)

python==3.5.6

matplotlib==2.2.2

numpy==1.15.2

pandas==0.23.4

astropy==3.0.4

scipy==1.3.0

# Input Data
You need to provide a file with a list of IDs for each source to be
analyzed.  It is assumed to be a single column with no filename.

Running convertPhotoLC requires a data table for each source (named
photoBase + ID + photoEnd where ID is the number listed in
the above ID list file as specified in OzDES_makeLC_run.py)
with four labeled columns: MJD, MAG, MAGERR, BAND.

Running makePhotoLC, makeLineLC, calcWidth, and calcBH required
calibrated spectroscopic data (spectraBase + ID + spectraEnd).  This
data is assumed to be of the form obtained after running
[OzDES_calibSpec](https://github.com/jhoormann/OzDES_calibSpec).  If
your data is in another form modify the SpectrumCoadd class defined in
OzDES_makeLC_calc.py to handle your data.

If you choose to run makePhotoLC you will need to provide the
photometric transmission function for the bands you want to calculate.
Provide the transmission function for each of the filters in a two
column format: wavelength (nm) and transmission fraction (range 0-1).

# Output Data

# Reference
If you are using this code please cite the paper where this procedure
was first presented and link to this github repository,
[Hoormann et al 2019, MNRAS 487, 3:3650](https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.3650H/abstract).
