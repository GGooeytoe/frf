from typing import List,Tuple
import numpy as np
from numpy.typing import ArrayLike
from timeit import default_timer
from fbs import block_thomas,conventional
from matplotlib import pyplot

def get_block_receptances_at_frequency(frequency:float,nominal_receptance:ArrayLike,
                                       configbandgaps:List[List[Tuple[float,float,ArrayLike]]]):
    '''
    assumes multiple configs, each with same nominal receptance across all frequencies except within specified bandgaps.

    Parameters: frequency : float
                    the frequency to query the matrices at
                nominal_receptance : (n,n) float array
                    the nominal receptance outside of the bandgaps (same for all frequencies and configs)
                configbandgaps : list[(float,float,(n,n) float array))]
                    list of lists of the bandgaps in each config. Outer list is the configs, inner list is bandgaps for that config. Each entry is (starting frequency, ending frequency, receptance in gap)
    '''
    Y=[]
    for config in configbandgaps:
        Yi=None
        for low,high,strength in config:
            if low<=frequency and frequency<=high:
                Yi=strength
                break
        if Yi is None:
            Yi=nominal_receptance
        Y.append(Yi)
    return Y

n=2#degrees of freedom in each unit cell
nominal_receptance=np.diag(n*[0.95])
config1=[(100,1000,np.diag(n*[0.5]))]
config2=[(2000,10000,np.diag(n*[0.5]))]
frequencies=[10**(.1*count) for count in range(60)]
recipe=[0,0]
N=len(recipe)#number of unit cells
excitation=np.array([1 for _ in range(n)]+[0 for _ in range(n*(N-1))])#unit excitation on each DOF of first cell, zero elsewhere

Yis=[get_block_receptances_at_frequency(f,nominal_receptance,[config1,config2]) for f in frequencies]
thomas_start=default_timer()
usthomas=np.array([block_thomas.compute_overall_frf(Yis[i],recipe,excitation) for i in range(len(Yis))])
thomas_end=default_timer()
usconventional=np.array([conventional.compute_overall_frf(Yis[i],recipe,excitation) for i in range(len(Yis))])
conventional_end=default_timer()
print(f"{N} unit cells, {n} cell DOF, {len(frequencies)} frequencies. Thomas={thomas_end-thomas_start:.4f}s, Conventional={conventional_end-thomas_end:.4f}s")
fig=pyplot.figure()
ax=pyplot.axes()
for i in range(n*N):
    ax.semilogx(frequencies,usthomas[:,i],":o",label=f"block Thomas [{i}]")
    ax.semilogx(frequencies,usconventional[:,i],":x",label=f"Conventional [{i}]")
ax.set_xlabel("Frequency")
ax.set_ylabel("Gain")
ax.legend()
pyplot.show()
