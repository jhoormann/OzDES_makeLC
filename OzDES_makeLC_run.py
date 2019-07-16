# ---------------------------------------------------------- #
# ------------------- OzDES_makeLC_run.py ------------------ #
# -------- https://github.com/jhoormann/OzDES_makeLC ------- #
# ---------------------------------------------------------- #
# This code was written for the OzDES Reverberation Mapping  #
# Program to create the emission line light curves.  This    #
# code reads in photometric and spectroscopic data.  It will #
# convert the photometric magnitudes into fluxes and extract #
# line fluxes from the spectroscopic data.  In order to      #
# isolate the emission lines the continuum is subtracted by  #
# a local estimation of a linear continuum.  You are also    #
# able to measure the line widths and estimate black hole    #
# mass.  The bulk of the calculations are performed in       #
# OzDES_makeLC_calc.py.  Unless otherwise noted this code    #
# was written by Janie Hoormann.                             #
# ---------------------------------------------------------- #

import numpy as np
import OzDES_makeLC_calc as ozcalc


# The name of the file that contains the ID for each source to be analyzed.  It is assumed to be a single column
# with no column name
sourceStatsFilename = "input/names.txt"

sourceStats = np.genfromtxt(sourceStatsFilename, 'U')
nSources = len(sourceStats)  # number of sources to loop over

# Define the pre/post necessary to get the spectroscopic data.  It assumes the data is named in the following
# convention: spectraBase + ID + spectraEnd where ID is the number listed in sourceStatsFilename.
# It is assumed that the spectral data is saved as a fits file in the form that the OzDES_calibSpec code outputs
# when coadding on date/run.  The data is read in with the SpectrumCoadd class.  If your data is in a different format
# you will need to modify this class to suit your needs.
spectraBase = "input/"
spectraEnd = "_scaled.fits"

# Define the pre/post necessary to get the photometric data.  It assumes the data is named in the following
# convention: photoBase + ID + photoEnd where ID is the number listed in sourceStatsFilename.
# It is assumed that the data is in the following format, as required by the rest of the OzDES scripts.
# 4 labeled columns (MJD, MAG, MAGERR, BAND :  these column names are assumed below).
photoBase = "input/"
photoEnd = "_lc.dat"

# Define where you want any output from the code to be saved to
outLoc = "output/"

# Flags to specify what you want the code to do
makeFig = True  # make/save figures for each AGN including coadd spectrum and light curves
makeFigEpoch = False  # make/save figures to illustrate spectrum for each epoch
convertPhotoLC = False  # convert/save photometric lightcurves converting from magnitudes to flux
makePhotoLC = False  # make light curves by applying photometric filters to spectral data
makeLineLC = False  # make/save line lightcurves
calcWidth = True  # calculate emission line widths
calcBH = True  # calculate the black hole mass from R-L relationship

# Define the emission lines you want to study (makeLineLC = True)
lineName = np.array(['CIV', 'MgII', 'Hbeta', 'Halpha'])  # line names
lineLoc = np.array([1549, 2798, 4861, 6563])  # rest frame line location in Angstroms
lumLoc = np.array([1350, 3000, 5100, 5100])  # the rest frame wavelengths that correspond to the luminosities used for
# each emission line in the R-L relationship
lineInt = {'CIV': [1470, 1595], 'MgII': [2700, 2920], 'Hbeta': [4810, 4940], 'Halpha': [6420, 6680]}  # the integration
# window for each emission line

# Define the relevant windows for continuum subtraction (in the rest frame)
# The local continuum subtraction windows on both sides of the emission line
contWinMin = {'CIV': [1450, 1460], 'MgII': [2190, 2210], 'Hbeta': [4760, 4790], 'Halpha': [6190, 6210]}
contWinMax = {'CIV': [1780, 1790], 'MgII': [3007, 3027], 'Hbeta': [5100, 5130], 'Halpha': [6810, 6830]}

# In order to calculate the uncertainties associated with continuum subtraction I move the continuum subtraction
# windows around to determine how much of an impact their choice has on the line flux.  The window is moved within these
# ranges.
contWinBSMin = {'CIV': [1435, 1480], 'MgII': [2180, 2240], 'Hbeta': [4700, 4800], 'Halpha': [6120, 6220]}
contWinBSMax = {'CIV': [1695, 1820], 'MgII': [2987, 3057], 'Hbeta': [5080, 5180], 'Halpha': [6800, 6900]}

# Photometric bands to be considered when converting photometric light curves (convertPhotoLC = True)
bandName = np.array(['g', 'r', 'i'])  # photometric band names, same band names used in the photometric data file
bandPivot = np.array([4812, 6434, 7815])  # pivot wavelength for each band

# If you want to make a lightcurve using photometric filters from spectral data define the bands and the transmission
# functions you want to use (makePhotoLC == True).  I am not going to check here, make sure the bands overlap yourself!
# You need bandName and bandPivot as defined above to match these filters.  The transmission function for each of the
# photometric filters in a two column format: wavelength (nm) and transmission fraction (range 0-1)
filters = {'g': '../OzDES_Pipeline/RMPipeline/input/DES_g_y3a1.dat',
           'r': '../OzDES_Pipeline/RMPipeline/input/DES_r_y3a1.dat',
           'i': '../OzDES_Pipeline/RMPipeline/input/DES_i_y3a1.dat'}

# The OzDES fluxes are on the order of 10^-16 ergs/s/cm^2/A.  To make it prettier to plot define a constant to scale
# the numbers by
scale = pow(10, -17)

# I use a bootstrap resampling technique to get uncertainties for continuum subtraction and line widths.  strapNum
# defines the number of resamplings to perform
strapNum = 100

for source in sourceStats:

    # Read in the spectral data if it will be needed later
    if True in [makeLineLC, makePhotoLC, calcWidth, calcBH]:
        specName = spectraBase + source + spectraEnd
        spectra = ozcalc.SpectrumCoadd(specName)
        # name some convenience variables.  If you needed to edit the SpectrumCoadd class to handle your spectroscopic
        # data format make sure the following data is readily available.  The fluxes/variances are scaled
        wavelength = spectra.wavelength
        origFluxes = spectra.flux/scale
        origVariances = spectra.variance/pow(scale, 2)
        origFluxCoadd = spectra.fluxCoadd/scale
        origVarCoadd = spectra.varianceCoadd/pow(scale, 2)
        z = spectra.redshift
        dates = spectra.dates
        numEpochs = spectra.numEpochs
        nBins = len(wavelength)

        # decide which emission lines are available in the spectrum
        availLines = ozcalc.findLines(wavelength, z, lineName, contWinBSMin, contWinBSMax)

    # Make the emission line light curves for all available emission lines
    if makeLineLC == True:
        ozcalc.lineLC(dates, lineName, availLines, lineInt, contWinMin, contWinMax, contWinBSMin, contWinBSMax,
                      wavelength, origFluxes, origVariances, origFluxCoadd, numEpochs, scale, z, strapNum, outLoc,
                      source, makeFig, makeFigEpoch)

    # Convert photometric lightcurves from magnitudes to flux and save as separate light curves
    if convertPhotoLC == True:
        # Define photometric filename
        photoName = photoBase + source + photoEnd
        ozcalc.convertPhotoLC(photoName, source, bandName, bandPivot, scale, makeFig, outLoc)

    # Make the light curves for a specified photometric filter given a series of spectra
    if makePhotoLC == True:
        ozcalc.makePhotoLC(dates, bandName, bandPivot, filters, wavelength, origFluxes, origVariances, numEpochs, scale,
                           outLoc, source, makeFig)

    # Now we will calculate the emission line width, while we are at it we will calculate the black hole mass if
    # possible.  Here I will just use the R-L relationships.  However, you could use these light curves to calculate
    # lags (using PyCCF, Javelin, etc) and read those lags in here for each line.  That part is up to you!  I will
    # calculate FWHM and velocity dispersion for the mean and RMS spectra.  It also corrects for the resolution of the
    # spectrograph assuming the red/blue arms of AAOmega.  If you are using a different setup change the resolutions
    # and splice locations in the findRes function.  If you want to use any other emission lines besides Hbeta, MgII,
    # and CIV you will need to add the R-L relationships
    if calcWidth == True:
        wave = wavelength/(1+z)
        ozcalc.calcWidth(wavelength, lineName, lineLoc, availLines, lineInt, lumLoc, contWinMin, contWinMax,
                         contWinBSMin, contWinBSMax, origFluxes, origVariances, origFluxCoadd, origVarCoadd, z,
                         strapNum, scale, outLoc, source, makeFig, calcBH)


