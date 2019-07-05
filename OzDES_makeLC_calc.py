# ---------------------------------------------------------- #
# ------------------ OzDES_makeLC_calc.py ------------------ #
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
# mass.  The functions are called in OzDES_makeLC_run.py.    #
# Unless otherwise noted this code was written by            #
# Janie Hoormann.                                            #
# ---------------------------------------------------------- #

from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d


# -------------------------------------------------- #
# Modified from a function originally provided by    #
# Anthea King                                        #
# -------------------------------------------------- #
# ----------------- SpectrumCoadd ------------------ #
# -------------------------------------------------- #
# Read in calibrated spectral data assuming data is  #
# in the format provided by OzDES_calibSpec after    #
# coadding.                                          #
# -------------------------------------------------- #

class SpectrumCoadd(object):
    # Spectrum class for latest version of the OzDES pipeline

    def __init__(self, filepath=None):
        assert filepath != None, "No file name is specified."
        self.filepath = filepath
        try:
            self.data = fits.open(filepath)
        except IOError:
            print("Error: file {0} could not be found".format(filepath))
            exit()
        data = fits.open(filepath)
        self.combined = data[0]
        self.combinedVariance = data[1]
        self._wavelength = None
        self._flux = None
        self._variance = None
        self._fluxCoadd = None
        self._varianceCoadd = None
        self._dates = None
        self._runs = None
        self.numEpochs = int((np.size(data) - 3) / 3)
        self.redshift = self.combined.header['z']
        self.RA = self.combined.header['RA']
        self.DEC = self.combined.header['DEC']
        self.field = self.combined.header['FIELD']


    @property
    def wavelength(self):
        """Define wavelength solution."""
        if getattr(self, '_wavelength', None) is None:
            crpix = self.combined.header[
                        'crpix1'] - 1.0  # Central pixel value. The -1.0 is needed as Python is ZERO indexed
            crval = self.combined.header['crval1']  # central wavelength value
            self.cdelt = self.combined.header['cdelt1']  # Wavelength interval between subsequent pixels
            n_pix = self.combined.header["NAXIS1"]
            wave = ((np.arange(n_pix) - crpix) * self.cdelt) + crval
            self._wavelength = wave
        return self._wavelength

    @property
    def flux(self):
        if getattr(self, '_flux', None) is None:
            self._flux = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._flux[:, i] = self.data[i * 3 + 3].data
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 3 + 4].data
        return self._variance

    @property
    def fluxCoadd(self):
        if getattr(self, '_fluxCoadd', None) is None:
            self._fluxCoadd = np.zeros(5000, dtype=float)
            self._fluxCoadd[:] = self.data[0].data
        return self._fluxCoadd

    @property
    def varianceCoadd(self):
        if getattr(self, '_varianceCoadd', None) is None:
            self._varianceCoadd = np.zeros(5000, dtype=float)
            self._varianceCoadd[:] = self.data[1].data
        return self._varianceCoadd

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = self.data[i * 3 + 3].header[
                    'AVGDATE']  # this give the average Modified Julian Date (UTC) that observation was taken
        return self._dates

    @property
    def runs(self):
        if getattr(self, '_runs', None) is None:
            self._runs = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._runs[i] = self.data[i * 3 + 3].header['RUN']  # this give the run number of the observation
        return self._runs


# -------------------------------------------------- #
# ------------------- magToFlux -------------------- #
# -------------------------------------------------- #
# Reads in magnitude, error, and pivot wavelength    #
# and converts to f_lambda in units of ergs/s/cm^2/A #
# -------------------------------------------------- #

def magToFlux(mag, err, pivot):
    flux = (3*pow(10,18)/pow(pivot,2))*pow(10, -(2./5.)*(mag + 48.6))
    flux_err = abs(flux*(-2./5.)*2.30259*err)
    return flux, flux_err


# -------------------------------------------------- #
# ------------------- outputLC --------------------- #
# -------------------------------------------------- #
# Creates an output file with date, flux, error      #
# columns as is expected by lag recovery tools       #
# Javelin and PyCCF.                                 #
# -------------------------------------------------- #
def outputLC(date, flux, error, name, loc, obj_name):

    length = len(date)
    outname = loc + obj_name + "_" + name + ".txt"
    output = open(outname, 'w')

    for i in range(length):
        if np.isnan(flux[i]) == False:
            output.write("%s    %s    %s \n" % (date[i], flux[i], error[i]))
        else:
            # Sometimes the flux ends up as nan, this is generally because the SNR is so bad/the emission line so
            # small that the continuum subtraction makes the line negative.  These are not saved in the data file
            # but a warning is outputted so you can have a look at what the problem is.
            print("-------\n  Houston, we have a problem! " + obj_name + " Night " + str(i) + "\n-------\n ")

    output.close()

    return


# -------------------------------------------------- #
# ------------------ plot_fonts -------------------- #
# -------------------------------------------------- #
# Function to define the font used for plotting.     #
# -------------------------------------------------- #
def plot_fonts(size, color='black', weight='normal', align='bottom'):
    font = {'size': size, 'color': color, 'weight': weight, 'verticalalignment': align}
    return font


# -------------------------------------------------- #
# ------------------ plot_ticks ---------------------#
# -------------------------------------------------- #
# Function to change the plot tick size.             #
# -------------------------------------------------- #
def plot_ticks(ax, size):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(size)
    return


# -------------------------------------------------- #
# ----------------- plot_share_x --------------------#
# -------------------------------------------------- #
# Define figure and axis variables for plot which    #
# shares the x axis for a specified number of plots. #
# -------------------------------------------------- #
def plot_share_x(number, title, xlabel, ylabel, xlim=(0, 0), ylim=(0, 0), asize=22, tsize=22, xdim=10,
                 ydim=10, xtick=5, ytick=5):
    fig, ax_array = plt.subplots(number, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(xdim, ydim, forward=True)
    fig.subplots_adjust(hspace=0)

    title_font = plot_fonts(tsize, align='bottom')
    x_axis_font = plot_fonts(asize, align='top')
    y_axis_font = plot_fonts(asize, align='bottom')

    for [i, ax] in enumerate(ax_array):
        plot_ticks(ax, asize)
        ax.set_ylabel(ylabel[i], **y_axis_font)
        ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))
        ax.tick_params(axis='y', pad=15)
        if ylim != (0, 0) and ylim[0] < ylim[1]:
            ax.set_ylim(ylim[i])
        if i == 0:
            ax.set_title(title, **title_font)
            ax.tick_params(axis='x', pad=15)
            if xlim != (0, 0) and xlim[0] < xlim[1]:
                ax.set_xlim(xlim)
        if i == number - 1:
            ax.set_xlabel(xlabel, **x_axis_font)
        if ytick is not None:
            ax.yaxis.major.locator.set_params(nbins=ytick)
        if xtick is not None:
            ax.xaxis.major.locator.set_params(nbins=xtick)

    return fig, ax_array

# -------------------------------------------------- #
# ------------------- photoLC -----------------------#
# -------------------------------------------------- #
# Define figure and axis variables for plot which    #
# shares the x axis for a specified number of plots. #
# -------------------------------------------------- #
def photoLC(photoName, source, bandName, bandPivot, scale, makeFig, outLoc):
    # Read in the photometric data
    photo = pd.read_table(photoName, delim_whitespace=True)

    if makeFig == True:
        # Define figure and axis for light curves of all bands
        fig_photo, ax_photo = plot_share_x(len(bandName), source, "Date (MJD)", bandName)

    # Make a light curve for each band
    for b in range(len(bandName)):
        # Create an array for observations of a specified band and sort observations by date
        band_data = photo[photo['BAND'] == bandName[b]].sort_values('MJD')
        # Find date, mag, and magerr array for the specified band
        ph_date = np.array(band_data['MJD'])
        ph_mag = np.array(band_data['MAG'])
        ph_magerr = np.array(band_data['MAGERR'])

        # Loop over each epoch and convert magnitude to flux
        ph_flux = np.zeros(len(ph_date))
        ph_fluxerr = np.zeros(len(ph_date))

        for e in range(len(ph_date)):
            ph_flux[e], ph_fluxerr[e] = magToFlux(ph_mag[e], ph_magerr[e], bandPivot[b])

        # Scale the fluxes before they are saved, if you are concerned about remembering the scale factor perhaps
        # included it in the outputted file name.
        ph_flux = ph_flux / scale
        ph_fluxerr = ph_fluxerr / scale

        # Save the data as a light curve with filename outLoc + source + _ + bandName[b] + .txt
        outputLC(ph_date, ph_flux, ph_fluxerr, bandName[b], outLoc, source)

        if makeFig == True:
            # plot the light curve on the subplot defined above.
            ax_photo[b].errorbar(ph_date, ph_flux, yerr=ph_fluxerr, fmt='o', color='black')

    # Once all the light curves are plotted save the figure as outLoc + source + "_photo.png"
    if makeFig == True:
        fig_photo.savefig(outLoc + source + "_photo.png")

    return


# -------------------------------------------------- #
# ------------------ findLines ----------------------#
# -------------------------------------------------- #
# Determines which emission lines are present in the #
# spectrum.  Returns an array of booleans where True #
# means the emission line is present.                #
# -------------------------------------------------- #
def findLines(wavelength, z, lineName, contWinBSMin, contWinBSMax):
    # decide which emission lines are available in the spectrum
    availLines = np.zeros(len(lineName)).astype(bool)

    for l in range(len(lineName)):
        # for a line to be in the spectrum you need to include the continuum subtraction windows as well.  This can
        # be limiting but as we need continuum subtracted spectra it is necessary.
        minWave = min(contWinBSMin[lineName[l]])
        maxWave = max(contWinBSMax[lineName[l]])

        if minWave * (1 + z) > wavelength[0] and maxWave * (1 + z) < wavelength[-1]:
            availLines[l] = True

    return availLines


# -------------------------------------------------- #
# -------------------- findBin ----------------------#
# -------------------------------------------------- #
# Finds the bin of the given vector (wavelength)     #
# where the specified quantity (line) is located.    #
# -------------------------------------------------- #
def findBin(line, wavelength):
    bin = 0
    for i in range(len(wavelength)-1):
        if line >= wavelength[i] and line <= wavelength[i+1]:
            bin = i
            i = len(wavelength)
        if line > wavelength[-1]:
            bin = len(wavelength)-1
            i = len(wavelength)
    return bin


# -------------------------------------------------- #
# ---------------- interpolateVals ------------------#
# -------------------------------------------------- #
# Interpolates a linear line between two points and  #
# propagates the uncertainty.                        #
# -------------------------------------------------- #
def interpolateVals(x, y, s, val):
    # uncertainty is variance

    interp = y[0] + (val - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    interp_var = s[0] + (s[0] + s[1]) * ((val - x[0]) / (x[1] - x[0])) ** 2.

    return interp, interp_var


# -------------------------------------------------- #
# ------------------ meanUncert ---------------------#
# -------------------------------------------------- #
# Finds the uncertainty corresponding to the mean    #
# of a set of numbers.                               #
# -------------------------------------------------- #
def meanUncert(variance):
    length = len(variance)
    var = 0
    num = 0
    for i in range(length):
        if np.isnan(variance[i]) == False:
            var = var + variance[i]
            num += 1

    sigma2 = (var / (num ** 2))

    return sigma2


# -------------------------------------------------- #
# ---------------- cont_fit_reject ------------------#
# -------------------------------------------------- #
# Interpolate a linear line through the mean of the  #
# continuum subtraction windows to represent the     #
# continuum and subtract this line.  Modifies the    #
# given flux/variance vectors.                       #
# -------------------------------------------------- #
def cont_fit_reject(wavelength, fluxes, variances, minWin, maxWin):

    # Define the wavelength range for the continuum model, between the mean of both windows
    wave = np.array([np.nanmean(minWin), np.nanmean(maxWin)])
    nBins = len(wavelength)

    # Determine how many epochs there are to continuum subtract
    number = int(fluxes.size / nBins)

    for epoch in range(number):
        if number == 1:
            flux = fluxes
            variance = variances
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        # Calculate the average flux at each extreme of the wave vector (ie mean of the continuum subtraction window)
        fvals = np.array([np.nanmean(flux[findBin(minWin[0], wavelength):findBin(minWin[1], wavelength)]),
                          np.nanmean(flux[findBin(maxWin[0], wavelength):findBin(maxWin[1], wavelength)])])

        # Calculate the average uncertainty at each extreme of the wave vector
        svals = np.array([meanUncert(variance[findBin(minWin[0], wavelength):findBin(minWin[1], wavelength)]),
                          meanUncert(variance[findBin(maxWin[0], wavelength):findBin(maxWin[1], wavelength)])])

        cont = np.zeros(nBins)
        contVar = np.zeros(nBins)

        # Find the interpolated linear continuum model
        for i in range(nBins):
            cont[i], contVar[i] = interpolateVals(wave, fvals, svals, wavelength[i])

        # Subtract the continuum from the flux and add the error of the model in quadrature with the spectral error
        flux -= cont
        variance += contVar

    return


# -------------------------------------------------- #
# ----------------- computeABmag ------------------- #
# -------------------------------------------------- #
# computes the AB magnitude for given transmission   #
# functions and spectrum (f_lambda).  Returns the    #
# magnitude and variance.                            #
# -------------------------------------------------- #
def computeABmag(trans_flux, trans_wave, tmp_wave, tmp_flux, tmp_var):
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(trans_wave)
    if minV < min(tmp_wave):
        minV = min(tmp_wave)
    maxV = max(trans_wave)
    if maxV > max(trans_wave):
        maxV = max(trans_wave)

    interp_wave = []
    tmp_flux2 = []
    tmp_var2 = []

    # Make new vectors for the flux just using that range (assuming spectral binning)

    for i in range(len(tmp_wave)):
        if minV < tmp_wave[i] < maxV:
            interp_wave.append(tmp_wave[i])
            tmp_flux2.append(tmp_flux[i])
            tmp_var2.append(tmp_var[i])

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_flux2 = interp1d(trans_wave, trans_flux)(interp_wave)

    # And now calculate the magnitude and uncertainty

    c = 2.992792e18  # Angstrom/s
    Num = np.nansum(tmp_flux2 * trans_flux2 * interp_wave)
    Num_var = np.nansum(tmp_var2 * (trans_flux2 * interp_wave) ** 2)
    Den = np.nansum(trans_flux2 / interp_wave)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
            magABvar = 1.17882 * Num_var / (Num ** 2)
        except FloatingPointError:
            magAB = 99.
            magABvar = 99.

    return magAB, magABvar

# --------------------------------------------------- #
# --------------- uncertainty_cont ------------------ #
# --------------------------------------------------- #
# This function finds the uncertainty in line flux    #
# and width measurements.  For line flux you can      #
# input a range of potential continuum windows and    #
# it will randomly pick regions to use for continuum  #
# subtraction. You can also input a region over which #
#  to randomly choose the integration window.  These  #
# all also include flux randomization in order to     #
# consider the effect of the variance spectrum.       #
# You can also look at the effect flux randomization  #
# has on the line width measurements FWHM and         #
# velocity dispersion.  You can also specify to look  #
# at the RMS spectrum (flag='rms') for the line width #
# measurements, the default is to look at the provided#
# spectrum as is.  The error is calculated through    #
# bootstrap resampling using strapNum iterations.     #
# The standard deviation of the calculated quantity   #
# is then the associated error.                       #
# --------------------------------------------------- #
def uncertainty_cont(wavelength, flux, variance, strapNum, z, line, pivotLC, winLimMin, winLimMax, winsize, scale,
                     calc='cont', flag='mean'):

    # calc = cont -> continuum subtraction
    # calc = win -> integration window
    # calc = fwhm -> FWHM line width: can specify flag=rms
    # calc = sigma -> line velocity dispersion: can specify flag=rms

    # Performs bootstrap resampling in the range of potentially clean continuum to determine
    # uncertainties on the flux measurement

    # Continuum window in Angstroms - will be scaled according to redshift

    # Winsize means the continuum subtraction windows are all the same size, just the locations shift
    winsize = winsize/(1+z)

    lineMin = line[0]
    lineMax = line[1]

    # Option for potentially clean continuum region pass in bootstrap

    # Calculate the width of the bootstrapping region on each side of the line
    lowW = (winLimMin[1]-winLimMin[0])/(1+z)
    highW = (winLimMax[1]-winLimMax[0])/(1+z)

    # Check edge conditions: if the bootstraping region goes outside the region of the spectrograph use the spectrograph
    # bounds as the edges
    if winLimMin[0] < wavelength[0]:
        winLimMin[0] = wavelength[0]
        winLimMin[1] = (winLimMin[0] / (1 + z) + lowW) * (1 + z)
    if winLimMin[1] > wavelength[line[0]]:
        winLimMin[1] = wavelength[line[0]]
    if winLimMax[1] > wavelength[-1]:
        winLimMax[1] = wavelength[-1]
        winLimMax[0] = (winLimMax[1] / (1 + z) - highW) * (1 + z)
    if winLimMax[0] < wavelength[line[1]]:
        winLimMax[0] = wavelength[line[1]]


    # Wavelengths to choose in each window in steps of 0.5A
    winMinVect = np.arange(winLimMin[0], winLimMin[1] - (winsize - 0.5) * (1 + z), 0.5 * (1 + z))
    winMaxVect = np.arange(winLimMax[0], winLimMax[1] - (winsize - 0.5) * (1 + z), 0.5 * (1 + z))

    # Array of random continuum window starting points
    randVectMin = len(winMinVect) * np.random.rand(strapNum)
    randVectMin = randVectMin.astype(int)

    randVectMax = len(winMaxVect) * np.random.rand(strapNum)
    randVectMax = randVectMax.astype(int)

    # An array of values obtained through bootstrapping to determine uncertainties
    vals = np.zeros(strapNum)

    for i in range(strapNum):

        if calc == 'win':
            # subtracts from standard continuum but changes integration window, in this case feed in potential
            # integration windows instead of bootstrapping regions

            lineMinNew = findBin(winMinVect[randVectMin[i]], wavelength)
            lineMaxNew = findBin(winMaxVect[randVectMax[i]], wavelength)

            # Performs flux resampling to account for variance spectrum.  Flux values shifted by Gaussian scaled by
            # variance
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)

            # Continuum Subtract this new vector
            cont_fit_reject(wavelength, fluxC, varC, winLimMin, winLimMax)

            # Calculate the flux
            lc_mag, lc_mag_err = computeABmag(np.ones(len(wavelength[lineMinNew:lineMaxNew])),
                                              wavelength[lineMinNew:lineMaxNew], wavelength[lineMinNew:lineMaxNew],
                                              fluxC[lineMinNew:lineMaxNew]*scale, varC[lineMinNew:lineMaxNew]*
                                              pow(scale,2))

            vals[i], lc_mag_err = magToFlux(lc_mag, lc_mag_err**0.5, pivotLC)

        if calc == "cont":
            # changes cont region
            minWin = [winMinVect[randVectMin[i]], winMinVect[randVectMin[i]] + winsize * (1 + z)]
            maxWin = [winMaxVect[randVectMax[i]], winMaxVect[randVectMax[i]] + winsize * (1 + z)]

            # Performs flux resampling to account for variance spectrum.  Flux values shifted by Gaussian scaled by
            # variance
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)

            # Continuum Subtract this new vector
            cont_fit_reject(wavelength, fluxC, varC, minWin, maxWin)

            # Calculate the flux
            lc_mag, lc_mag_err = computeABmag(np.ones(len(wavelength[lineMin:lineMax])),wavelength[lineMin:lineMax],
                                              wavelength[lineMin:lineMax], fluxC[lineMin:lineMax]*scale,
                                              varC[lineMin:lineMax]*pow(scale, 2))

            vals[i], lc_mag_err = magToFlux(lc_mag, lc_mag_err**0.5, pivotLC)

        if calc == "fwhm":
            # Determine uncertainty in FWHM line measurement
            # do flux randomization and continuum subtraction
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)
            cont_fit_reject(wavelength, fluxC, varC, winLimMin, winLimMax)

            if flag == 'rms':
                # first calculate the RMS spectrum if requested
                fluxC, varC = rmsSpec(fluxC, varC)

            vals[i] = fwhm(wavelength[lineMin:lineMax], fluxC[lineMin:lineMax])

        if calc == "sigma":
            # Determine uncertainty in velocity dispersion measurement
            # do flux randomization and continuum subtraction
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)
            cont_fit_reject(wavelength, fluxC, varC, winLimMin, winLimMax)

            if flag == 'rms':
                # first calculate the RMS spectrum if requested
                fluxC, varC = rmsSpec(fluxC, varC)
            vals[i] = lineVar(wavelength[lineMin:lineMax], fluxC[lineMin:lineMax])

    stddev_bs = np.nanstd(vals)
    return stddev_bs


# --------------------------------------------------- #
# ----------------------- fwhm ---------------------- #
# --------------------------------------------------- #
# Takes an input spectrum and calculate the FWHM of   #
# the provided emission line.  It will search over    #
# the entire provided wavelength window so just       #
# include the relevant region of the spectrum.        #
# --------------------------------------------------- #
def fwhm(wave, flux, plot=False):

    # Find the half maximum
    peak = max(flux)
    valley = min(flux)
    peakLoc = wave[np.where(flux == peak)[0][0]]
    peakLocB = findBin(peakLoc, wave)
    hm = (peak-valley) / 2 + valley

    leftUp = wave[0]
    leftDown = wave[peakLocB]
    rightUp = wave[-1]
    rightDown = wave[peakLocB]

    # First search for the half max to the left of the line
    for i in range(peakLocB):
        # First search going up the line
        if flux[i] < hm < flux[i+1]:
            leftUp = (wave[i] + wave[i+1])/2
        # Then going down the line
        if flux[peakLocB-i-1] < hm < flux[peakLocB-i]:
            leftDown = (wave[peakLocB-i-1] + wave[peakLocB-i])/2

    # Then take the average which will account for any double peaks/noise in the spectrum
    left = (leftUp + leftDown)/2

    # And now to the right
    maxSize = len(wave) - 1

    for i in range(maxSize - peakLocB):
        # Go up
        if flux[peakLocB + i + 1] < hm < flux[peakLocB + i]:
            rightDown = (wave[peakLocB + i] + wave[peakLocB + i + 1])/2
        # Go down
        if flux[maxSize-i] < hm < flux[maxSize-i-1]:
            rightUp = (wave[maxSize-i] + wave[maxSize-i-1])/2

    right = (rightUp + rightDown)/2

    if plot == True:
        fig, ax = makeFigSingle("", "Wavelength", "Flux", [wave[0], wave[-1]])
        ax.plot(wave, flux, color='black', alpha=0.5)
        ax.plot([0, 10000], [hm, hm], color='black')
        ax.axvline(x=peakLoc, color='red')
        ax.axvline(x=left, color='blue')
        ax.axvline(x=right, color='blue')
        plt.show()

    # Now calculate the velocity

    # km/s
    c = 299792.458

    zLeft = (left-peakLoc)/peakLoc
    zRight = (right-peakLoc)/peakLoc

    zComb = (1+zRight)/(1+zLeft)-1

    vel = c*((1+zComb)**2-1)/((1+zComb)**2+1)

    return vel


# --------------------------------------------------- #
# ---------------------- lineVar -------------------- #
# --------------------------------------------------- #
# Takes an input spectrum and calculate the velocity  #
# dispersion of the emission line.  It will search    #
# over the entire provided wavelength window so just  #
# include the relevant region of the spectrum.        #
# --------------------------------------------------- #
def lineVar(wave, flux):
    length = len(wave)

    peak = max(flux)
    peakLoc = wave[np.where(flux == peak)[0][0]]

    # Calculate velocity dispersion following equation written in Peterson 2004, the three following constants
    # correspond to the main terms in that equation.

    Pdl = 0
    lPdl = 0
    l2Pdl = 0

    for i in range(length):
        Pdl += flux[i]

        lPdl += flux[i] * wave[i]

        l2Pdl += flux[i] * pow(wave[i], 2)

    lambda0 = lPdl / Pdl

    lambda2 = l2Pdl / Pdl

    lambda02 = pow(lambda0, 2)

    linevar = lambda2 - lambda02

    sigma = linevar ** 0.5

    c = 299792.458

    left = peakLoc - sigma / 2
    right = peakLoc + sigma / 2

    zLeft = (left - peakLoc) / peakLoc
    zRight = (right - peakLoc) / peakLoc

    # redshift from lambda_l to lambda_r
    zComb = (1 + zRight) / (1 + zLeft) - 1

    # convert to velocity
    vel = c * ((1 + zComb) ** 2 - 1) / ((1 + zComb) ** 2 + 1)

    return vel


# --------------------------------------------------- #
# --------------- exponential_smooth ---------------- #
# --------------------------------------------------- #
# Function to apply an exponential smoothing kernel   #
# to the data.  Written by Harry Hobson.              #
# --------------------------------------------------- #
def exponential_smooth(fluxes):

    number = int(fluxes.size/fluxes.shape[0])

    search_pixels = 5
    decay = 0.9

    window = np.arange(-search_pixels, search_pixels + 1)
    weights = decay ** np.abs(window)
    weights /= np.sum(weights)

    if (number == 1):
        flux = fluxes[:]
        flux[:] = np.convolve(flux, weights, mode='same')
    else:
        for epoch in range(fluxes.shape[1]):
            flux = fluxes[:, epoch]
            flux[:] = np.convolve(flux, weights, mode='same')


# --------------------------------------------------- #
# -------------------- meanSpec --------------------- #
# --------------------------------------------------- #
# Calculates the mean of multiple spectra as well as  #
# the corresponding variance spectrum.                #
# --------------------------------------------------- #
def meanSpec(flux, variance):

    length = len(flux[:,0])

    meanFlux = np.zeros(length)
    meanVar = np.zeros(length)

    for i in range(length):
        meanFlux[i] = np.nanmean(flux[i,:])
        meanVar[i] = np.nanmean(variance[i,:])

    return meanFlux, meanVar


# --------------------------------------------------- #
# -------------------- rmsSpec ---------------------- #
# --------------------------------------------------- #
# Calculates the RMS of the inputted spectra.  Will   #
# expect fluxes in [wavelength, epoch] format.  An    #
# exponential smoothing function is applied to the    #
# data as a first and last step to mitigate some of   #
# the noise.                                          #
# --------------------------------------------------- #
def rmsSpec(flux, variance):
    # smooth the input spectra
    exponential_smooth(flux)

    length = len(flux[:, 0])
    epochs = len(flux[0, :])

    # Calculate the RMS spectrum, variance propagated through but not used later
    mean, meanVar = meanSpec(flux, variance)
    rms = np.zeros(length)
    rmsVar = np.zeros(length)
    rmsVar2 = np.zeros(length)

    for b in range(length):
        for e in range(epochs):
            rms[b] += (flux[b, e] - mean[b]) ** 2
            rmsVar2[b] += 4 * rms[b] * (variance[b, e] + meanVar[b])
        rms[b] = (rms[b] / (epochs - 1)) ** 0.5
        rmsVar2[b] = rmsVar2[b] / ((epochs - 1) ** 2)
        rmsVar[b] = rmsVar2[b] * (0.5 / rms[b]) ** 2

    # smooth the final RMS spectrum
    exponential_smooth(rms)

    return rms, rmsVar


# -------------------------------------------------- #
# -------------------- lineLC ---------------------- #
# -------------------------------------------------- #
# Create emission line light curves by integrating   #
# the emission lines after local continuum           #
# subtraction.  The uncertainties due to the variance#
# of the spectrum and the continuum subtraction is   #
# performed through bootstrap resampling.  This is   #
# done for every emission line from the provided list#
# that is present in the spectrum.                   #
# -------------------------------------------------- #
def lineLC(dates, lineName, availLines, lineInt, contWinMin, contWinMax, contWinBSMin, contWinBSMax, wavelength,
           origFluxes, origVariances, numEpochs, scale, z, strapNum, outLoc, source, makeFig, makeFigEpoch):

    for l in range(len(lineName)):
        if availLines[l] == True:
            line = lineName[l]

            # Copy the flux/variance vectors so you have an uncontinuum subtracted version to use for other lines
            fluxes = np.copy(origFluxes)
            variances = np.copy(origVariances)

            # define some variables for line/continuum windows in observed frame
            contMin = np.array(contWinMin[line]) * (1 + z)
            contMax = np.array(contWinMax[line]) * (1 + z)
            contMinBS = np.array(contWinBSMin[line]) * (1 + z)
            contMaxBS = np.array(contWinBSMax[line]) * (1 + z)

            # similar for the line integration window but I want the wavelength bin number, not just the wavelength
            lineMin = findBin(lineInt[line][0] * (1 + z), wavelength)
            lineMax = findBin(lineInt[line][1] * (1 + z), wavelength)

            # Perform the continuum subtraction
            cont_fit_reject(wavelength, fluxes, variances, contMin, contMax)

            lc_mag = np.zeros(numEpochs)
            lc_mag_sigma = np.zeros(numEpochs)
            lc_flux = np.zeros(numEpochs)
            lc_flux_sigma = np.zeros(numEpochs)
            total_error = np.zeros(numEpochs)

            # Calculate the pivot wavelength associated with each line window
            pivotLC = pow(np.nansum(wavelength[lineMin:lineMax]) / np.nansum(1 / wavelength[lineMin:lineMax]), 0.5)

            # Calculate magnitudes and fluxes for each line
            for epoch in range(numEpochs):
                # first calculate magnitudes, save these if you want to compare this instead of fluxes
                # Here the transmission function is 1 for all wavelengths within the integration window.
                lc_mag[epoch], lc_mag_sigma[epoch] = computeABmag(np.ones(len(wavelength[lineMin:lineMax])),
                                                                  wavelength[lineMin:lineMax],
                                                                  wavelength[lineMin:lineMax],
                                                                  fluxes[lineMin:lineMax, epoch] * scale,
                                                                  variances[lineMin:lineMax, epoch] * pow(scale, 2))
                # Now convert to flux, this is what is saved.  Note: all fluxes here are actually flux densities
                # This uncertainty just considers the variance spectrum, we will take everything in the next step
                lc_flux[epoch], lc_flux_sigma[epoch] = magToFlux(lc_mag[epoch], lc_mag_sigma[epoch] ** 0.5, pivotLC)
                total_error[epoch] = uncertainty_cont(wavelength, origFluxes[:, epoch], origVariances[:, epoch],
                                                      strapNum, z, [lineMin, lineMax], pivotLC, contMinBS,
                                                      contMaxBS, contMin[1] - contMin[0], scale)

            # Scale the line fluxes as with the photometry
            lc_flux = lc_flux / scale
            total_error = total_error / scale

            # Save the data as a light curve with filename outLoc + source + _ + line + .txt
            outputLC(dates, lc_flux, total_error, line, outLoc, source)

    return


# -------------------------------------------------- #
# ----------------- makeFigSingle ------------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A function that defines a figure with legible axis #
# labels.                                            #
# -------------------------------------------------- #
font = {'size': '20', 'color': 'black', 'weight': 'normal'}
def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)

    ax = fig.add_subplot(111)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    ax.set_ylabel(ylabel, **font)
    if ylim != [0, 0] and ylim[0] < ylim[1]:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax.set_xlim(xlim)

    ax.set_title(title, **font)

    return fig, ax

