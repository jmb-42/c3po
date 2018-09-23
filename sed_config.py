# Functions for SED fitting
import os
import pickle
from time import time
from string import ascii_uppercase as UPPERS
import numpy as np
from scipy import integrate
from scipy import constants
from scipy.interpolate import interp1d as interp
from scipy.interpolate import UnivariateSpline as uni_spline
import matplotlib.pyplot as plt
from astropy.io import ascii

################################################################################
####                          BEGIN OPTIONS
################################################################################

# These should be the only directories you have to change.
HOME_DIR = '/Users/lacc/Documents/Justin/c3po/'
FPATH = HOME_DIR + 'Temperatures/'
ARR_DIR = HOME_DIR + 'Arrays/'

GRAINSIZES = np.loadtxt(ARR_DIR+'GrainSizes.dat')
WAVELENGTHS = np.loadtxt(ARR_DIR+'Wavelengths.dat')
STAR_TEMPS = np.linspace(2000, 15000, 14)
DISK_RADII = np.logspace(-1, 3, 121)
WAVES = np.logspace(-1, 3, 1000)
grainComps = ['AstroSil', 'DirtyIceAstroSil']
GRAIN_TEMPS_TOTAL = dict()
EMISSIVITIES_TOTAL = dict()
for grain in grainComps:
    GRAIN_TEMPS_TOTAL[grain] = np.load(ARR_DIR+grain+'GrainTemps.npy')
    EMISSIVITIES_TOTAL[grain] = np.load(ARR_DIR+grain+'Emissivities.npy')

################################################################################
####                           END OPTIONS
################################################################################

# Find nearest value in an array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Convolution function for calibrating the IRS data
def convolve(filterwaves, filterresponse, datawaves, dataflux):
    top = integrate.simps(
      np.interp(filterwaves, datawaves, dataflux)*filterresponse,
        filterwaves)
    bottom = integrate.simps(filterresponse, filterwaves)
    return top/bottom

def stitch_trim_lr(sData, spitzFlags, spitzNames):
    wa = 'wavelength'
    fx = 'flux'
    er = 'error'
    sl2f, sl1f, ll2f, ll1f = spitzFlags
    sl2, sl1, ll2, ll1 = spitzNames

    # Before stitching, we trim spitzer data to <35 microns
    if ll1f and 0:
        index = np.where(sData[ll1][wa]<35.0)
        sData[ll1][fx] = sData[ll1][fx][index]
        sData[ll1][er] = sData[ll1][er][index]
        sData[ll1][wa] = sData[ll1][wa][index]

    k=3
    lastvalue = -2
    firstvalue = 1
    if sl1f and sl2f:
        sl1half = sData[sl1][wa].size/2
        sl2half = sData[sl2][wa].size/2
        SL1_fit = uni_spline(sData[sl1][wa][:sl1half],
            sData[sl1][fx][:sl1half],k=k)
        SL2_fit = uni_spline(sData[sl2][wa][sl2half:],
            sData[sl2][fx][sl2half:],k=k)
        SL2_SL1 = np.linspace(sData[sl2][wa][lastvalue],
            sData[sl1][wa][firstvalue], 3)
        SL2_N = np.nanmean(SL2_fit(SL2_SL1))
        SL1_N = np.nanmean(SL1_fit(SL2_SL1))

        SL1_norm = SL2_N / SL1_N
        sData[sl1][fx] *= SL1_norm

    if ll2f and sl1f:
        ll2half = sData[ll2][wa].size/2
        sl1half = sData[sl1][wa].size/2
        LL2_fit = uni_spline(sData[ll2][wa][:ll2half],
            sData[ll2][fx][:ll2half],k=k)
        SL1_fit = uni_spline(sData[sl1][wa][sl1half:],
            sData[sl1][fx][sl1half:],k=k)
        SL1_LL2 = np.linspace(sData[sl1][wa][lastvalue],
            sData[ll2][wa][firstvalue], 3)
        LL2_N = np.nanmean(LL2_fit(SL1_LL2))
        SL1_N = np.nanmean(SL1_fit(SL1_LL2))

        LL2_norm = SL1_N / LL2_N
        sData[ll2][fx] *= LL2_norm

    if ll1f and ll2f:
        ll1half = sData[ll1][wa].size/2
        ll2half = sData[ll2][wa].size/2
        LL2_fit = uni_spline(sData[ll2][wa][ll2half:],
            sData[ll2][fx][ll2half:],k=k)
        LL1_fit = uni_spline(sData[ll1][wa][:ll1half],
            sData[ll1][fx][:ll1half],k=k)
        LL2_LL1 = np.linspace(sData[ll2][wa][lastvalue],
        sData[ll1][wa][firstvalue], 3)
        LL2_N = np.nanmean(LL2_fit(LL2_LL1))
        LL1_N = np.nanmean(LL1_fit(LL2_LL1))

        LL1_norm = LL2_N / LL1_N
        sData[ll1][fx] *= LL1_norm
    return sData

def stitch_lr(sData, spitzFlags, spitzNames):
    wa = 'wavelength'
    fx = 'flux'
    sl2f, sl1f, ll2f, ll1f = spitzFlags
    sl2, sl1, ll2, ll1 = spitzNames

    lastvalue = -5
    firstvalue = 4
    if sl1f and sl2f:
        sl1half = sData[sl1][wa].size/2
        sl2half = sData[sl2][wa].size/2

        SL1_fit = uni_spline(sData[sl1][wa][:sl1half],
            sData[sl1][fx][:sl1half])
        SL2_fit = uni_spline(sData[sl2][wa][sl2half:],
            sData[sl2][fx][sl2half:])

        # SL1_fit = uni_spline(sData[sl1][wa],
        #     sData[sl1][fx])
        # SL2_fit = uni_spline(sData[sl2][wa],
        #     sData[sl2][fx])

        mid = np.nanmean([sData[sl1][wa][firstvalue], sData[sl2][wa][lastvalue]])
        SL2_N = SL2_fit(mid)
        SL1_N = SL1_fit(mid)
        SL1_norm = SL2_N/SL1_N
        sData[sl1][fx] *= SL1_norm

    if ll2f and sl1f:
        ll2half = sData[ll2][wa].size/2
        sl1half = sData[sl1][wa].size/2

        LL2_fit = uni_spline(sData[ll2][wa][:ll2half],
            sData[ll2][fx][:ll2half])
        SL1_fit = uni_spline(sData[sl1][wa][sl1half:],
            sData[sl1][fx][sl1half:])

        # LL2_fit = uni_spline(sData[ll2][wa],
        #     sData[ll2][fx])
        # SL1_fit = uni_spline(sData[sl1][wa],
        #     sData[sl1][fx])

        mid = np.nanmean([sData[sl1][wa][firstvalue], sData[sl2][wa][lastvalue]])
        LL2_N = LL2_fit(mid)
        SL1_N = SL1_fit(mid)
        LL2_norm = SL1_N/LL2_N
        sData[ll2][fx] *= LL2_norm

    if ll1f and ll2f:
        ll1half = sData[ll1][wa].size/2
        ll2half = sData[ll2][wa].size/2

        LL2_fit = uni_spline(sData[ll2][wa][ll2half:],
            sData[ll2][fx][ll2half:])
        LL1_fit = uni_spline(sData[ll1][wa][:ll1half],
            sData[ll1][fx][:ll1half])

        # LL2_fit = uni_spline(sData[ll2][wa],
        #     sData[ll2][fx])
        # LL1_fit = uni_spline(sData[ll1][wa],
        #     sData[ll1][fx])


        mid = np.nanmean([sData[ll2][wa][firstvalue], sData[ll1][wa][lastvalue]])
        LL2_N = LL2_fit(mid)
        LL1_N = LL1_fit(mid)
        LL1_norm = LL2_N/LL1_N
        sData[ll1][fx] *= LL1_norm
    return sData

def stitch_trim_rl(sData, spitzFlags, spitzNames):
    wa = 'wavelength'
    fx = 'flux'
    er = 'error'
    sl2f, sl1f, ll2f, ll1f = spitzFlags
    sl2, sl1, ll2, ll1 = spitzNames

    # Before stitching, we trim spitzer data to <35 microns
    if ll1f:
        index = np.where(sData[ll1][wa]<35.0)
        sData[ll1][fx] = sData[ll1][fx][index]
        sData[ll1][er] = sData[ll1][er][index]
        sData[ll1][wa] = sData[ll1][wa][index]

    lastvalue = -4
    firstvalue = 3
    if ll2f and ll1f:
        ll1half = sData[ll1][wa].size/2
        ll2half = sData[ll2][wa].size/2
        LL2_fit = uni_spline(sData[ll2][wa][ll2half:],
            sData[ll2][fx][ll2half:])
        LL1_fit = uni_spline(sData[ll1][wa][:ll1half],
            sData[ll1][fx][:ll1half])
        LL2_LL1 = np.linspace(sData[ll2][wa][lastvalue],
        sData[ll1][wa][firstvalue], 100)
        LL2_N = np.nanmean(LL2_fit(LL2_LL1))
        LL1_N = np.nanmean(LL1_fit(LL2_LL1))
        LL2_norm = LL1_N/LL2_N
        sData[ll2][fx] *= LL2_norm
    if sl1f and ll2f:
        ll2half = sData[ll2][wa].size/2
        sl1half = sData[sl1][wa].size/2
        LL2_fit = uni_spline(sData[ll2][wa][:ll2half],
            sData[ll2][fx][:ll2half])
        SL1_fit = uni_spline(sData[sl1][wa][sl1half:],
            sData[sl1][fx][sl1half:])
        SL1_LL2 = np.linspace(sData[sl1][wa][lastvalue],
            sData[ll2][wa][firstvalue], 100)
        LL2_N = np.nanmean(LL2_fit(SL1_LL2))
        SL1_N = np.nanmean(SL1_fit(SL1_LL2))
        SL1_norm = LL2_N/SL1_N
        sData[sl1][fx] *= SL1_norm
    if sl2f and sl1f:
        sl1half = sData[sl1][wa].size/2
        sl2half = sData[sl2][wa].size/2
        SL1_fit = uni_spline(sData[sl1][wa][:sl1half],
            sData[sl1][fx][:sl1half])
        SL2_fit = uni_spline(sData[sl2][wa][sl2half:],
            sData[sl2][fx][sl2half:])
        SL2_SL1 = np.linspace(sData[sl2][wa][lastvalue],
            sData[sl1][wa][firstvalue], 100)
        SL2_N = np.nanmean(SL2_fit(SL2_SL1))
        SL1_N = np.nanmean(SL1_fit(SL2_SL1))
        SL2_norm = SL1_N/SL2_N
        sData[sl2][fx] *= SL2_norm
    return sData

def norm_blackbodies(sData, fitWaves, nWarm, nCold, t_1, t_2):

    wa = 'wavelength'
    fx = 'flux'
    er = 'error'
    w4 = 'WISE4'
    mp24, mp70 = 'MIPS24', 'MIPS70'
    h70, h100 = 'HerschelPACS70', 'HerschelPACS100'
    h160 = 'HerschelPACS160'
    ll1  = 'SpitzerIRS-LL1'

    # Which values are present in the data?
    mp24f, ll1f, w4f = nWarm
    mp70f, h70f, h100f, h160f = nCold

    # Initial guesses for the black bodies

    # fitWaves = [23.68, 22.194]
    # fitWaves += [i+1 for i in range(160)]
    # fitWaves = sorted(fitWaves)
    # bb1 = b_lam(fitWaves, t_1)
    # bb2 = b_lam(fitWaves, t_2)

    wav = np.logspace(-1, 2.3, 1000)
    bb1 = b_lam(wav, t_1)
    bb2 = b_lam(wav, t_2)



    # Normalize warm dust
    if mp24f:
        n_1 = np.nanmean(sData[mp24][fx]) / bb1.max()
    elif ll1f:
        index2 = np.where(np.logical_and(sData[ll1][wa]>20, sData[ll1][wa]<25))
        n_1 = np.nanmean(sData[ll1][fx][index2]) / bb1.max()
    elif w4f:
        n_1 = np.nanmean(sData[w4][fx]) / bb1.max()
    else:
        n_1 = 1

    # Normalize cold dust
    if mp70f:
        n_2 = np.nanmean(sData[mp70][fx]) / bb2.max()
    elif h70f:
        n_2 = np.nanmean(sData[h70][fx]) / bb2.max()
    elif h100f:
        n_2 = np.nanmean(sData[h100][fx]) / bb2.max()
    elif h160f:
        n_2 = np.nanmean(sData[h160][fx]) / bb2.max()
    else:
        n_2 = n_1

    # # Normalize warm dust
    # if mp24f:
    #     # index1 = np.where(fitWaves==23.68)
    #     index1 = np.where(np.logical_and(wav>=23, wav<=24))
    #     n_1 = np.nanmean(sData[mp24][fx]) / np.nanmean(bb1[index1])
    # elif ll1f:
    #     # index1 = np.where(np.logical_and(fitWaves>=20, fitWaves<=25))
    #     index1 = np.where(np.logical_and(wav>=20, wav<=25))
    #     index2 = np.where(np.logical_and(sData[ll1][wa]>20, sData[ll1][wa]<25))
    #     n_1 = np.nanmean(sData[ll1][fx][index2]) / np.nanmean(bb1[index1])
    # elif w4f:
    #     # index1 = np.where(fitWaves==22.194)
    #     index1 = np.where(np.logical_and(wav>=21, wav<=23))
    #     n_1 = np.nanmean(sData[w4][fx]) / np.nanmean(bb1[index1])
    # else:
    #     n_1 = 1

    # # Normalize cold dust
    # if mp70f:
    #     # index1 = np.where(fitWaves==71.42)
    #     index1 = np.where(np.logical_and(wav>=70.0, wav<=73.0))
    #     n_2 = np.nanmean(sData[mp70][fx]) / np.nanmean(bb2[index1])
    # elif h70f:
    #     # index1 = np.where(fitWaves==70.0)
    #     index1 = np.where(np.logical_and(wav>=69.0, wav<=71.0))
    #     n_2 = np.nanmean(sData[h70][fx]) / np.nanmean(bb2[index1])
    # elif h100f:
    #     # index1 = np.where(fitWaves==100.0)
    #     index1 = np.where(np.logical_and(wav>=98.0, wav<=102.0))
    #     n_2 = np.nanmean(sData[h100][fx]) / np.nanmean(bb2[index1])
    # elif h160f:
    #     # index1 = np.where(fitWaves==160.0)
    #     index1 = np.where(np.logical_and(wav>=150.0, wav<=170.0))
    #     n_2 = np.nanmean(sData[h160][fx]) / np.nanmean(bb2[index1])
    # else:
    #     n_2 = n_1
    return n_1, n_2

# Blackbody radiation function, in terms of lambda. Returns FNu in janskies
def b_lam(waves, temp):
    # Constants
    H = constants.h
    C = constants.c
    K = constants.k
    waves_m = waves/1e6
    return (2*(C**2)*H)/((np.exp((H*C)/(waves_m*K*temp))-1.0)*(waves_m**5)) \
      * (waves_m**2/C) * 1.0e19

# Blackbody radiation function, in terms of nu. Returns FNu in janskies
def b_nu(wavelengths, temperature):
    H = constants.h
    C = constants.c
    K = constants.k
    wavelengths_m = wavelengths/1.0e6
    nu = C/wavelengths_m
    top = 2 * H * (nu**3)
    bottom = C**2 * (np.exp((H*nu)/(K*temperature)) - 1)
    return top/bottom

def blowout_size(grainDensity, starL=1., starM=1., qRad=0.9):
    if starL is np.nan:
        starL = 1
    if starM is np.nan:
        starM = 1
    G = constants.G
    C = constants.c
    # Calculate blowout grain size
    grainDensity = grainDensity / 1000 / (0.01**3) # Density converted to SI
    nume = 3 * starL * 3.826e26 * qRad
    deno = 8 * np.pi * starM * 1.989e30 * G * C * grainDensity
    return nume/deno * 1e6

# Separates data according to instrument. Input: dict of IPAC Tables
def sort_by_instrument(data):
    wa = 'wavelength'
    fx = 'flux'
    er = 'error'
    unique_insts = list()
    for i in range(data['instrument'].size):
        if not data['instrument'][i] in unique_insts:
            unique_insts.append(data['instrument'][i])
    separatedStarData = dict()
    for inst in unique_insts:
        index = np.where(data['instrument']==inst)
        separatedStarData[inst] = dict()
        separatedStarData[inst][wa] = np.array(data[wa][index])
        separatedStarData[inst][fx] = np.array(data[fx][index])
        separatedStarData[inst][er] = np.array(data[er][index])
    return unique_insts, separatedStarData

# Remove .DS_Store file in directory
def pull_file_names(dirPath):
    filenames = os.listdir(dirPath)
    if '.DS_Store' in filenames:
        os.remove(dirPath+'.DS_Store')
        ind = filenames.index('.DS_Store')
        filenames.pop(ind)
    return filenames

# Create 1000 radii arrays for given star temp. Used in realistic fitting.
# Takes dictionary of grain temps and list of grain comps.
def interpTemps(starTemp, oldGrainTemps, grainComps):
    STAR_TEMPS = np.linspace(2000, 15000, 14)
    DISK_RADII = np.logspace(-1, 3, 121)
    radii = np.logspace(-1, 3, 1000)
    GRAINSIZES = np.loadtxt(HOME_DIR+'Arrays/GrainSizes.dat')
    for grainComp in grainComps:
        abr = ''
        for letter in grainComp:
            if letter in UPPERS:
                abr += letter
        starIndices = np.where(np.logical_and(
            STAR_TEMPS<starTemp+3000,
            STAR_TEMPS>starTemp-3000
            ))
        newStarTempGrainTemps = np.empty((
            DISK_RADII.size,
            GRAINSIZES.size
            ))
        for r in range(DISK_RADII.size):
            for g in range(GRAINSIZES.size):
                newStarTempGrainTemps[r][g] = np.interp(
                    starTemp,
                    STAR_TEMPS[starIndices],
                    oldGrainTemps[grainComp][starIndices][:,r][:,g]
                    )
        newGrainTemps = np.empty((radii.size,GRAINSIZES.size))
        for r  in range(radii.size):
            for g in range(GRAINSIZES.size):
                newGrainTemps[r][g] = np.interp(
                    radii[r],
                    DISK_RADII,
                    newStarTempGrainTemps[:,g]
                    )
        np.save(HOME_DIR+'Arrays/InterpGrainTemps/'+'%.0fK_%s.npy'%
            (starTemp,grainComp), newGrainTemps, allow_pickle=False)

class StarObject:
    def __init__(self, starD, starL, grainComp, grainTemps, blowoutSize, emis, grains):
        self.starD = starD
        self.starL = starL
        self.grainComp = grainComp
        self.grainTemps = grainTemps
        self.blowoutSize = blowoutSize
        self.emis = emis
        self.grains = grains
        self.G = constants.G        # Universal gravitation constant
        self.C = constants.c        # Speed of light in vacuum
        self.S = constants.sigma    # Stephan-Boltzman constant
        self.K = constants.k        # Boltzman constant
        self.H = constants.h        # Planck's constant
        self.AS_DENSITY = 3.0     # g/cm3
        self.H2O_DENSITY = 1.0    # g/cm3  LATER: use vlfr to calc grain density
        self.AC_DENSITY  = 2.095  # g/cm3
        self.radii = np.logspace(-1, 3, 1000)

    def calcFlux(self, waves, r0, T_0=1):
        wavelengths_m = waves / 1.0e6
        # Create radii/grains arrays
        sigma = 0.10 # Use 0.10 for the deviation
        r0 /= np.sqrt(self.starL)
        rindex = np.where(np.logical_and(self.radii<1.4*r0,
            self.radii>0.6*r0))[0]
        radii = self.radii[rindex]
        radii *= 1.4959787066e11
        r0 *= 1.4959787066e11
        grainTemps = self.grainTemps[rindex]
        grains = self.grains/1.0e6

        # Calculate CA
        # T_0 = 1
        blS = self.blowoutSize/1e6
        q = -3.5
        exponent = -0.5 * ((radii - r0) / (sigma*r0))**2
        ca = T_0*np.exp(exponent)*np.abs(3+q) \
             / (np.pi*(np.power(blS,3+q)-np.power(.001,3+q)))
        ca *= 1e6

        # Integral loop
        da = np.diff(grains)
        da = np.append(da, da[-1])
        dr = np.diff(radii)
        dr = np.append(dr, dr[-1])
        fw = np.empty(waves.size)
        fr = np.empty(radii.size)

        flux = np.empty(grains.size)
        for w in range(waves.size):
            for r in range(radii.size):
                B_nu = b_nu(waves[w], grainTemps[r])
                grainflux = (grains**2/(4*((self.starD*3.08568025e16)**2))) \
                    * self.emis[:,w] * B_nu * ca[r] * (grains**-3.5) * da \
                    * 2 * np.pi * radii[r] * dr[r]
                fr[r] = grainflux.sum()
            fw[w] = fr.sum()*1e26
        return fw

if __name__ == '__main__':
    before = time()
    densities = {
        'AstroSil': 3.0,
        'DirtyIceAstroSil': 1.12
        }

    asdf = ascii.read(HOME_DIR+'stars final sorted/star_HD 32977.txt')
    WAVES = np.array(sorted(asdf['wavelength']))
    starD = asdf.meta['keywords']['DIST_pc']['value']
    if starD is np.nan:
        starD = 1
    starT = asdf.meta['keywords']['TEMP']['value']
    if starT is np.nan:
        starT = 5800
    WAVES = np.logspace(-1,3, 901)

    starL = 1
    r0 = 0.5
    T_0 = 1
    grainComp = 'AstroSil'
    blowoutSize = blowout_size(densities[grainComp])

    try:
        grainTemps = np.load(HOME_DIR+'Arrays/InterpGrainTemps/'+'%.0fK_%s.npy'%(
            starT, grainComp))
    except:
        from sed_config import GRAIN_TEMPS_TOTAL
        interpTemps(starT, GRAIN_TEMPS_TOTAL, [grainComp])
        del GRAIN_TEMPS_TOTAL
        grainTemps = np.load(HOME_DIR+'Arrays/InterpGrainTemps/'+'%.0fK_%s.npy'%(
            starT, grainComp))

    graindex = np.where(GRAINSIZES>=blowoutSize)[0]
    grains = GRAINSIZES[graindex]
    grainTemps = grainTemps[:,graindex]

    # Interp emissivities to wavelengths
    emis = np.empty((grains.size, WAVES.size))
    for g in range(grains.size):
        emis[g] = np.interp(WAVES,WAVELENGTHS,EMISSIVITIES_TOTAL[grainComp][g])

    # (waves, r0, starD, grainComp, grainTemps, blowoutSize, emis, grains)
    star = StarObject(starD, grainComp, grainTemps, blowoutSize,
        emis, grains)
    before = time()
    flux1 = star.calcFlux(WAVES, r0, blowoutSize)
    plt.plot(WAVES, flux1, label = 0.5)
    print time()-before
    flux2 = star.calcFlux(WAVES, 1., blowoutSize)
    plt.plot(WAVES, flux2, label = 1.)

    flux3 = star.calcFlux(WAVES, 5., blowoutSize)
    plt.plot(WAVES, flux3, label = 5.)

    flux4 = star.calcFlux(WAVES, 50., blowoutSize)
    plt.plot(WAVES, flux4, label = 50.)

    flux5 = star.calcFlux(WAVES, 200., blowoutSize)
    plt.plot(WAVES, flux5, label = 200.)

    bb1 = b_lam(WAVES, 200)
    index = np.where(np.logical_and(WAVES>=25., WAVES<=35.))
    bb1 *= np.nanmean(flux1[index])/np.nanmean(bb1[index])
    plt.plot(WAVES, bb1, '-.', label='BB')
    plt.legend()
    plt.ylim(1, 1e4)
    # plt.xlim(2, 200)
    # plt.xlim(0.1, 1000)
    # plt.ylim(1e-8, 1e16)
    plt.xlim(2, 1000)
    plt.semilogx()
    plt.semilogy()
    plt.show()
