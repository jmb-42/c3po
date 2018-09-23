# Realistic grain fitting routine.
# As of right now, this code only fits two belt debris disks. I haven't
# implemented the single belt fits yet.

import os
from collections import OrderedDict
from time import time
from scipy.interpolate import UnivariateSpline as uni_spline
from scipy.interpolate import interp1d as interp
from scipy import integrate
from scipy.optimize import curve_fit
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from sed_config import (find_nearest, convolve, pull_file_names, b_lam,
                        sort_by_instrument, WAVES, WAVELENGTHS,
                        ARR_DIR, blowout_size, interpTemps,
                        GRAINSIZES, StarObject, find_nearest_ind,
                        stitch_trim_lr, stitch_trim_rl)
from direcs import PATH, STAR_FILES, KURUCZ, NEXTGEN, IMG_DIR, INTERPS_DIR

################################################################################
####                          BEGIN OPTIONS
################################################################################

show = 1    # Show plots?
save = 0    # Save plots to IMG_DIR ?

# Grab file names from star files directory
starList = pull_file_names(STAR_FILES)
# Grab star names from file names
starNames = list()
for i in range(len(starList)):
    starNames.append(starList[i].rstrip('_stitched.txt'))


# NOTE: It's possible to override the code such that it doesn't iterate through
# all of the stars. We can choose just one star or any number of arbitrary
# stars to do the fit for. There's one stipulation:
# The name(s) of the star we want to do a fit for MUST be within a list, even if
# it's just a single star. This is because we iterate through the star list.
# Just make sure to keep the brackets around single stars as well as lists.
# There's two lines that serve as an example of how this works below.
# There should be a text file with all of the star names you have so that you
# can choose any of them individually if you so choose. This is best for
# debugging if necessary.

# starNames = ['HD 70313', 'HD 159492', 'HD 12039']
starNames = ['HD 3196']

################################################################################
####                           END OPTIONS
################################################################################


# Main loop definition.
def run_fits(starName):
    # Convenience variables
    wa = 'wavelength'
    fx = 'flux'
    er = 'error'
    kw = 'keywords'
    va = 'value'

    # Instrument names
    mj, mh, mk      = '2MASSJ', '2MASSH', '2MASSK'
    w1, w2, w3, w4  = 'WISE1', 'WISE2', 'WISE3', 'WISE4'
    mp24, mp70      = 'MIPS24', 'MIPS70',
    mp24ul, mp70ul  = 'MIPS24UL', 'MIPS70UL'
    h70, h100, h160 = 'HerschelPACS70', 'HerschelPACS100', 'HerschelPACS160'
    h70ul, h100ul   = 'HerschelPACS70UL', 'HerschelPACS100UL'
    h160ul          = 'HerschelPACS160UL'
    sl2, sl1        = 'SpitzerIRS-SL2', 'SpitzerIRS-SL1'
    ll2, ll1        = 'SpitzerIRS-LL2', 'SpitzerIRS-LL1'

    # Grain compositions. This directly impacts the files that are loaded.
    warmGrain = 'AstroSil'
    coldGrain = 'DirtyIceAstroSil'

    # Grain densities. Important for calculating the blowoutsize of the grains.
    densities = {
        # For the density of the grains:
        # Mixture of H20:NH3 is 3:1 and 10% vlfr of aC to make dirty ice.
        # then dirty ice is 50% vlfr of AstroSil to make DIAS
        # Assuming:
        # 1.0 g/cm^3 for H20
        # 0.817 g/cm^3 for NH3
        # 3.0 g/cm^3 for AstroSil
        'DirtyIce': 1.07,
        'AstroSil': 2.7,
        'DirtyIceAstroSil': 1.885
        # 'AstroSil': 3.0,           # Correlates to DIAS -> 2.034 g/cm^3
        # 'DirtyIceAstroSil': 2.034, # Correlates to AS -> 3.0 g/cm^3
        }

    # Read in the star file
    starData = ascii.read(STAR_FILES+'%s_stitched.txt'%starName)

    # Read in the stellar properties
    starT    = starData.meta[kw]['TEMP'][va]
    starL    = starData.meta[kw]['starL'][va]
    starM    = starData.meta[kw]['starM'][va]
    specType = starData.meta[kw]['SpType'][va]
    starD    = starData.meta[kw]['DIST_pc'][va]

    # Fillers for in case the data is missing.
    if np.isnan(starL):
        starL = 1
    if np.isnan(starM):
        starM = 1
    if np.isnan(starD):
        starD = 1

    # Concerning the nextgen and kurucz models: the wavelengths are in units of
    # angstrom, which equals 10^-10 meters. Flux is in erg/cm^2/s/a.
    # Create temp array that matches the temperatures of the nextgen stellar
    # models. From 2600-4100, uses a step of 100. From 4200-10000, uses a step
    # of 200. Grabs the nearest model to the star temp in the star file.
    ngTemps    = np.arange(2600, 4100, 100)
    ngTemps    = np.append(ngTemps, np.arange(4200, 10200, 200))
    TEMP       = find_nearest(ngTemps, starT)
    ngfilename = 'xp00_'+str(TEMP)+'g40.txt'

    # Temperature label for the graph of the SED fit
    starLabel = TEMP

    # Load the nexgen stellar model
    ngWave, ngFlux   = np.loadtxt(NEXTGEN+ngfilename, unpack=True)
    ngWave, ngwIndex = np.unique(ngWave, return_index=True)
    ngFlux           = ngFlux[ngwIndex]

    # Convert the nextgen model to janskies
    c_cgs = 2.99792458e10                          #cm/s
    ngFnu = ngFlux*(ngWave**2)/c_cgs/1.0e8*1.0e23  #Jy
    ngWave = ngWave*1.0e-4                         #cm -> um

    # If the star temperature exceeds the nextgen models max temp, then we
    # create a 'Frankenstein' stellar model. We combine flux from the kurucz
    # model with the flux from the highest nextgen stellar model
    if TEMP > 10000:
        # Load kurucz model. Similar to nextgen, the models are at specific
        # temperatures, with certain ranges being at different step sizes.
        kzTemps = np.arange(3500, 10000, 250)
        kzTemps = np.append(kzTemps, np.arange(10000, 13000, 500))
        kzTemps = np.append(kzTemps, np.arange(13000, 35000, 1000))
        kzTemps = np.append(kzTemps, np.arange(35000, 52500, 2500))
        kzTemp  = find_nearest(kzTemps, starT)

        # Replace temperature label for the graph of the SED fit
        starLabel = kzTemp

        kzfilename       = 'kp00_'+str(kzTemp)+'g40.txt'
        kzWave, kzFlux   = np.loadtxt(KURUCZ+kzfilename, unpack=True)
        kzWave, kzwIndex = np.unique(kzWave, return_index=True)
        kzFlux           = kzFlux[kzwIndex]

        # Convert the kurucz model to janskies
        c_cgs  = 2.99792458e10                          #cm/s
        kzFnu  = kzFlux*(kzWave**2)/c_cgs/1.0e8*1.0e23  #Jy
        kzWave = kzWave*1.0e-4                         #cm -> um

        # Isolate the kurucz flux to wavelengths less than 2.154
        index  = np.where(kzWave < 2.154)
        kzFnu  = kzFnu[index]
        kzWave = kzWave[index]

        # Isolate the nextgen flux to wavelengths greater than 2.154
        index  = np.where(ngWave >= 2.154)
        ngFnu  = ngFnu[index]
        ngWave = ngWave[index]

        # Normalize the data (stitch it together)
        s_norm = np.nanmean(kzFnu[-5:])/np.nanmean(ngFnu[:5])
        ngFnu *= s_norm

        ngWave = np.array(list(kzWave)+list(ngWave))
        ngFnu  = np.array(list(kzFnu)+list(ngFnu))

    # Log all instruments and sort data by instrument. sData is a dictionary,
    # the keys of which are the instrument names. Within that are the fluxes,
    # the wavelengths, and errors.
    insts, sData = sort_by_instrument(starData)

    # Boolean flags used to simplify accounting for which modules are present
    # in the data
    mjf, mhf, mkf    = mj in insts, mh in insts, mk in insts
    w1f, w2f         = w1 in insts, w2 in insts
    w3f, w4f         = w3 in insts, w4 in insts
    sl2f, sl1f       = sl2 in insts, sl1 in insts
    ll2f, ll1f       = ll2 in insts, ll1 in insts
    mp24f, mp24ulf   = mp24 in insts, mp24ul in insts
    mp70f, mp70ulf   = mp70 in insts, mp70ul in insts
    h70f, h100f      = h70 in insts, h100 in insts
    h160f, h70ulf    = h160 in insts, h70ul in insts
    h100ulf, h160ulf = h100ul in insts, h160ul in insts

    # Iterator lists for the flags
    # NOTE: must be in matching order
    spitzFlags = [sl2f, sl1f, ll2f, ll1f]
    spitzNames = [sl2, sl1, ll2, ll1]
    instFlags  = [mjf,mhf,mkf,w1f,w2f,w3f,w4f,mp24f,mp70f,h70f,h100f,h160f]
    instNames  = [mj, mh, mk, w1, w2, w3, w4, mp24, mp70, h70, h100, h160]
    ulFlags    = [mp24ulf, mp70ulf, h70ulf, h100ulf, h160ulf]
    ulNames    = [mp24ul, mp70ul, h70ul, h100ul, h160ul]

    # Saturation limits for each instrument
    satLims = {mj:10.057, mh:10.24, mk:10.566, w1:0.18, w2:0.36, w3:0.88,
        w4:12.0, h70: 220., h100:510., h160:1125., mp24:np.inf, mp70:np.inf}

    # Create a list of instrument names for non-spitzer/non-upper limit data
    totalInsts = list()
    for i, f in enumerate(instFlags):
        if f:
            totalInsts.append(instNames[i])

    # Create a list of instrument names for upper limits data
    ulInsts = list()
    for i, f in enumerate(ulFlags):
        if f:
            ulInsts.append(ulNames[i])

    # Create a list of instruments names for spitzer data
    spitzInsts = list()
    for i, f in enumerate(spitzFlags):
        if f:
            spitzInsts.append(spitzNames[i])

    # Colors for plotting accessed by instrument name
    plotColors = {
        mj: 'r', mh: 'r', mk: 'r',
        w1: 'b', w2: 'b', w3: 'b', w4: 'b',
        mp24:   'g', mp70:   'g',
        mp24ul: 'g', mp70ul: 'g',
        h70:   'purple', h100:   'purple', h160:   'purple',
        h70ul: 'purple', h100ul: 'purple', h160ul: 'purple',
        }

    # Begin the arrays that will be used in the fitting optimization function.
    fitWaves = list()
    fitFlux  = list()
    fitError = list()
    if totalInsts:
        for inst in totalInsts:
            # if sData[inst][fx] < satLims[inst]: # Uncomment to exclude saturated vals
            if sData[inst][fx] < np.inf: # Uncomment to use all vals
                fitWaves += list(sData[inst][wa])
                fitFlux  += list(sData[inst][fx])
                fitError += list(sData[inst][er])
    if ulInsts:
        for inst in ulInsts:
            fitWaves += list(sData[inst][wa])
            fitFlux  += list(sData[inst][fx])
            fitError += list(sData[inst][er])

    # Create arrays for the stitched data
    if spitzInsts:
        spitzWaves = list()
        spitzFlux  = list()
        spitzError = list()
        for ins in spitzInsts:
            spitzWaves += list(sData[ins][wa])
            spitzFlux  += list(sData[ins][fx])
            spitzError += list(sData[ins][er])
        spitzWaves = np.array(spitzWaves)
        spitzFlux  = np.array(spitzFlux)
        spitzError = np.array(spitzError)

    # Do convolving if there's MIPS24 data and IRS
    if mp24f and spitzInsts:
        # Convolve IRS data to the MIPS24 data
        MIPS24W = sData[mp24][wa]
        MIPS24F = sData[mp24][fx]
        mipsw, mipsr = np.loadtxt(PATH+'/Filter Response/mips24_frf.txt',
            unpack=True)
        IRS24      = convolve(mipsw, mipsr, spitzWaves, spitzFlux)
        spitzFlux *= (MIPS24F/IRS24)
        fitWaves  += list(spitzWaves)
        fitFlux   += list(spitzFlux)
        fitError  += list(spitzError)

    # Else, just add the spitzer data
    elif not mp24f and spitzInsts:
        fitWaves += list(spitzWaves)
        fitFlux  += list(spitzFlux)
        fitError += list(spitzError)

    # Organize all data by increasing wavelength.
    sFitWaves = list()
    sFitFlux  = list()
    sFitError = list()
    for i in range(len(fitWaves)):
        index = fitWaves.index(min(fitWaves))
        sFitWaves.append(fitWaves[index])
        sFitFlux.append(fitFlux[index])
        sFitError.append(fitError[index])
        fitWaves.pop(index)
        fitFlux.pop(index)
        fitError.pop(index)
    fitWaves = np.array(sFitWaves)
    fitFlux  = np.array(sFitFlux)
    fitError = np.array(sFitError)

    # Normalize stellar model from either SL2 or 2MASSK data.
    if sl2f:
        ind          = np.where(np.logical_and(fitWaves>5, fitWaves<5.5))
        dataFluxNorm = np.nanmean(fitFlux[ind])
        ngFluxNorm   = np.nanmean(np.interp(fitWaves[ind], ngWave, ngFnu))
        n_3   = (dataFluxNorm/ngFluxNorm)
        ngFnu = n_3 * ngFnu
    elif mkf:
        dataFluxNorm = sData[mk][fx]
        ngFluxNorm   = np.nanmean(np.interp(sData[mk][wa], ngWave, ngFnu))
        n_3   = (dataFluxNorm/ngFluxNorm)
        ngFnu = n_3 * ngFnu
    else:
        print "There is no SL2 or 2MASSK data to normalize the stellar model."
        return 0        # Return is a way of exiting the function early

    # Interpolate the stellar model to the fitting data wavelengths
    ngFnu_fit = np.e**np.interp(np.log(fitWaves), np.log(ngWave), np.log(ngFnu))
    print 'Normalizing models to data/constructing initial params'

    # Create the star objects
    # Calculate blowoutsize given grain density
    blowoutSize1 = blowout_size(densities[warmGrain], starL, starM)
    blowoutSize2 = blowout_size(densities[coldGrain], starL, starM)

    # This part may require some experimentation to make the fit ideal.
    # blowoutSize1 *= 0.3 # HD 80950
    # blowoutSize2 *= 0.3 # HD 80950
    # blowoutSize1 *= 0.8
    # blowoutSize2 *= 0.8
    # blowoutSize1 *= 3
    # blowoutSize2 *= 3

    # Load the grain temperatures per grain composition
    grainTemps = dict()
    grainComps = ['AstroSil', 'DirtyIceAstroSil']
    # These files should be made already, but if not, then the script will
    # create them and then save them for subsequent fits.
    try:
        for grain in grainComps:
            grainTemps[grain] = np.load(
            INTERPS_DIR+'%.0fK_%s.npy'%(starT, grain))
    except:
        from sed_config import GRAIN_TEMPS_TOTAL
        interpTemps(starT, GRAIN_TEMPS_TOTAL, grainComps)
        for grain in grainComps:
            grainTemps[grain] = np.load(
            INTERPS_DIR+'%.0fK_%s.npy'%(starT, grain))

    # Limit grain arrays to greater than the blowout size
    graindex1 = np.where(GRAINSIZES>=blowoutSize1)[0]
    graindex2 = np.where(GRAINSIZES>=blowoutSize2)[0]
    grains1   = GRAINSIZES[graindex1]
    grains2   = GRAINSIZES[graindex2]
    grainTemps[warmGrain] = grainTemps[warmGrain][:,graindex1]
    grainTemps[coldGrain] = grainTemps[coldGrain][:,graindex2]

    # Grab the minimum radial location which that's below the temp
    # of sublimation for volatiles. (Icy belt radius)
    for r in range(1000):
        yellow = grainTemps[coldGrain][r]<120
        if not False in yellow:
            minRad = r
            break
    minRad = np.logspace(-1, 3, 1000)[minRad]

    print( '----------------------------------------' )
    print( '      AS blowout size: %.2f'  % blowoutSize1 )
    print( '     IMP blowout size: %.2f' % blowoutSize2 )
    print( '       r0 lower limit: %.2f'  % (0.5*starL**0.5) )
    print( ' Minimum radius for an icy belt: %.2f' % (minRad*np.sqrt(starL)) )

    # Load emissivities per grain comp (use these for hi res plots)
    TOTAL_EMISSIVITIES = dict()
    for grain in grainComps:
        TOTAL_EMISSIVITIES[grain] = np.load(ARR_DIR+grain+'Emissivities.npy')

    # Limit emissivities to grains greater than blowout size
    TOTAL_EMISSIVITIES[warmGrain] = TOTAL_EMISSIVITIES[warmGrain][graindex1,:]
    TOTAL_EMISSIVITIES[coldGrain] = TOTAL_EMISSIVITIES[coldGrain][graindex2,:]

    # Interp emissivities to fitWaves (use these for fitting)
    emis1 = np.empty((grains1.size, fitWaves.size))
    for g in range(grains1.size):
        emis1[g] = np.interp(fitWaves, WAVELENGTHS,
            TOTAL_EMISSIVITIES[warmGrain][g])
    emis2 = np.empty((grains2.size, fitWaves.size))
    for g in range(grains2.size):
        emis2[g] = np.interp(fitWaves, WAVELENGTHS,
            TOTAL_EMISSIVITIES[coldGrain][g])

    # Construct star objects (both fitting and hi res)
    star1 = StarObject(starD, starL, warmGrain, grainTemps[warmGrain],
        blowoutSize1, emis1, grains1)
    star2 = StarObject(starD, starL, coldGrain, grainTemps[coldGrain],
        blowoutSize2, emis2, grains2)
    star1_hi_res = StarObject(starD, starL, warmGrain, grainTemps[warmGrain],
        blowoutSize1, TOTAL_EMISSIVITIES[warmGrain], grains1)
    star2_hi_res = StarObject(starD, starL, coldGrain, grainTemps[coldGrain],
        blowoutSize2, TOTAL_EMISSIVITIES[coldGrain], grains2)

    normWaves = fitWaves
    # Interp emissivities to normWaves (only use these for norm factors)
    # Do this because sat values might limit fitWaves theoretical flux
    emisB1 = np.empty((grains1.size, normWaves.size))
    for g in range(grains1.size):
        emisB1[g] = np.interp(normWaves, WAVELENGTHS,
            TOTAL_EMISSIVITIES[warmGrain][g])
    emisB2 = np.empty((grains2.size, normWaves.size))
    for g in range(grains2.size):
        emisB2[g] = np.interp(normWaves, WAVELENGTHS,
            TOTAL_EMISSIVITIES[coldGrain][g])

    # Normalization star models
    starB1 = StarObject(starD, starL, warmGrain, grainTemps[warmGrain],
        blowoutSize1, emisB1, grains1)
    starB2 = StarObject(starD, starL, coldGrain, grainTemps[coldGrain],
        blowoutSize2, emisB2, grains2)

    # Calculate flux to normalize
    bbr1 = max(minRad*np.sqrt(starL)*0.5, 1*np.sqrt(starL))
    bbr2 = max(minRad*np.sqrt(starL)*1.5, 35.) # I DON'T REMEMBER WHY I PUT 35
    bb1 = starB1.calcFlux(normWaves, bbr1)
    bb2 = starB2.calcFlux(normWaves, bbr2)

    # Normalize warm dust
    if mp24f:
        # index1 = np.where(np.logical_and(normWaves>=23, normWaves<=24))
        # n_1 = np.nanmean(sData[mp24][fx]) / np.nanmean(bb1[index1])
        n_1 = np.nanmean(sData[mp24][fx]) / bb1.max()
    elif ll1f:
        # index1 = np.where(np.logical_and(normWaves>=20, normWaves<=25))
        index2 = np.where(np.logical_and(sData[ll1][wa]>20, sData[ll1][wa]<25))
        # n_1 = np.nanmean(sData[ll1][fx][index2]) / np.nanmean(bb1[index1])
        n_1 = np.nanmean(sData[ll1][fx][index2]) / bb1.max()
    elif w4f:
        # index1 = np.where(np.logical_and(normWaves>=21, normWaves<=23))
        # n_1 = np.nanmean(sData[w4][fx]) / np.nanmean(bb1[index1])
        n_1 = np.nanmean(sData[w4][fx]) / bb1.max()
    else:
        n_1 = 1

    # Normalize cold dust
    if mp70f:
        # index1 = np.where(np.logical_and(normWaves>=70.0, normWaves<=73.0))
        # n_2 = np.nanmean(sData[mp70][fx]) / np.nanmean(bb2[index1])
        n_2 = np.nanmean(sData[mp70][fx]) / bb2.max()
    elif h70f:
        # index1 = np.where(np.logical_and(normWaves>=69.0, normWaves<=71.0))
        # n_2 = np.nanmean(sData[h70][fx]) / np.nanmean(bb2[index1])
        n_2 = np.nanmean(sData[h70][fx]) / bb2.max()
    elif h100f:
        # index1 = np.where(np.logical_and(normWaves>=98.0, normWaves<=102.0))
        # n_2 = np.nanmean(sData[h100][fx]) / np.nanmean(bb2[index1])
        n_2 = np.nanmean(sData[h100][fx]) / bb2.max()
    elif h160f:
        # index1 = np.where(np.logical_and(normWaves>=150.0, normWaves<=170.0))
        # n_2 = np.nanmean(sData[h160][fx]) / np.nanmean(bb2[index1])
        n_2 = np.nanmean(sData[h160][fx]) / bb2.max()
    else:
        n_2 = n_1

    # Reset norm factor for ngfNu
    n_3 = 1

    # Define the optimization function
    def realistic_fit(waves, r0warm, r0cold, n1, n2, n3):
        return n1*star1.calcFlux(waves, r0warm) \
            + n2*star2.calcFlux(waves, r0cold) \
            + n3*ngFnu_fit

    try:
        print( '----------------------------------------' )
        print( '        optimizing parameters\n' )
        print( '     this might take a few minutes' )
        print( '----------------------------------------' )
        rw = bbr1
        rc = bbr2
        lBounds = [0.5*np.sqrt(starL), minRad*np.sqrt(starL),
                   n_1*1e-4, n_2*1e-4, n_3*0.8]
        uBounds = [minRad*np.sqrt(starL), 800.*np.sqrt(starL),
                   n_1*1e2, n_2*1e2, n_3*1.2]
        bounds =[lBounds, uBounds]

        # Timer for the routine
        before = time()
        # Call the optimization routine here.
        popt, pcov = curve_fit(
            realistic_fit, fitWaves, fitFlux, sigma=fitError,
            p0=(rw, rc, n_1, n_2, n_3),
            bounds=bounds
            )
        # Unpack the parameters
        RW, RC, n1, n2, n3 = popt

        print( '  Time to optimize parameters: %.2fs' % (time() - before) )
        print( '        Warm radius: %.2f' % RW )
        print( '        Cold radius: %.2f' % RC )
        print( '          Warm norm: %.2f' % n1 )
        print( '          Cold norm: %.2f' % n2 )
        print( '       Stellar norm: %.2f' % n3 )
        print( '----------------------------------------' )

        # Number of data points minus number of fitting parameters
        degsFreedom = fitWaves.size - 5
        resid = (fitFlux-realistic_fit(fitWaves, RW, RC, n1, n2, n3))/fitError
        chiSqr = np.dot(resid, resid)/degsFreedom # Reduced chi square

    except:
        print "Optimization failed for star %s" % starName
        plt.close('all')
        return 0        # Return is a way of exiting the function early



    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 18
    fig = plt.figure(figsize=(8,6))
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

    # hi res arrays for the optimized fluxes
    y1 = n1 * star1_hi_res.calcFlux(WAVELENGTHS, RW)
    y2 = n2 * star2_hi_res.calcFlux(WAVELENGTHS, RC)

    # hi res array for the total flux
    totalFlux = y1 + y2 + np.e**np.interp(np.log(WAVELENGTHS), np.log(ngWave),
        np.log(n3*ngFnu))

    # Plot realistic grain fluxes
    plt.plot(WAVELENGTHS, y1, ls='--', color='blue',
        label='Radial Location: %.2f AU'%RW)
    plt.plot(WAVELENGTHS, y2, ls='--', color='r',
        label='Radial Location: %.2f AU'%RC)

    # Plot stellar model, total flux, and IRS data
    plt.plot(ngWave, n3*ngFnu, color = 'gray',
        label='Next Gen T: %i K'%starLabel)
    plt.plot(WAVELENGTHS, totalFlux, color='lime',
        label='Total Flux')
    if spitzInsts:
        plt.plot(spitzWaves, spitzFlux, color='black',
            label='Spitzer IRS')

    # Plot the rest of the real data and error bars
    # We don't want distinct labels for upper limits, so we use this labels
    # list to check if those instruments have already been plotted.
    labels = []
    if totalInsts:
        for inst in totalInsts:
            label = inst.rstrip('0123456789JHK')
            if not label in labels:
                labels.append(label)
            else:
                label = None
            plt.scatter(sData[inst][wa], sData[inst][fx],
                s=20, marker='D', color=plotColors[inst], label=label,
                zorder=10)
            plt.errorbar(sData[inst][wa], sData[inst][fx],
                yerr=sData[inst][er], color=plotColors[inst])
    if ulInsts:
        for inst in ulInsts:
            label = inst.rstrip('0123456789JHKUL')
            if not label in labels:
                labels.append(label)
            else:
                label = None
            plt.scatter(sData[inst][wa], 3*sData[inst][er]+sData[inst][fx],
                s=20, marker='D', color=plotColors[inst], label=label,
                zorder=10)
            plt.errorbar(sData[inst][wa], 3*sData[inst][er]+sData[inst][fx],
                yerr=sData[inst][er], uplims=True, color=plotColors[inst])

    # Plot formatting
    plt.title(starName, fontsize=22)
    plt.xlabel(r'$\lambda$ ($\mu m$)')
    plt.ylabel(r'$F_{\nu}$ ($Jy$)')
    plt.semilogx()
    plt.semilogy()

    # Calculate the y limits based on the data
    if y1.max() < fitFlux[np.where(fitFlux>0)].min() or \
        y2.max() < fitFlux[np.where(fitFlux>0)].min():
        lowerylimit = min([y1.max(), y2.max()])*0.5
    else:
        lowerylimit = fitFlux[np.where(fitFlux>0)].min()*0.5

    if y1.max() > fitFlux.max() or y2.max() > fitFlux.max():
        upperylimit = max([y1.max(), y2.max()])*1.5
    else:
        upperylimit = fitFlux.max()*1.5
    plt.ylim(lowerylimit, upperylimit)

    # The x limits are standard
    # Sometimes, though, we might have to move around the height of the
    # chi square value on the plot because of the legend covering it.
    plt.xlim(.5,300)
    plt.text(.55, lowerylimit*1.2, r' Reduced $\chi ^2$: %.1f'
        %chiSqr)
    plt.legend()
    if save:
        plt.savefig(IMG_DIR+starName+'.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')

# Run the code!
for starName in starNames:
    run_fits(starName)
