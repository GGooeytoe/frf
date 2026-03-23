from typing import List,Tuple
import numpy as np
from timeit import default_timer
from fbs import block_thomas,conventional
from matplotlib import pyplot

def get_block_receptances_at_frequency(frequency:float,nominal_receptance:float,
                                       configbandgaps:List[List[Tuple[float,float,float]]]):
    '''
    assumes multiple configs, each with same nominal receptance across all frequencies except within specified bandgaps.

    Parameters: frequency : float
                    the frequency to query the matrices at
                nominal_receptance : float
                    the nominal receptance outside of the bandgaps (same for all frequencies and configs)
                configbandgaps : list[(float,float,float)]
                    list of lists of the bandgaps in each config. Outer list is the configs, inner list is bandgaps for that config. Each entry is (starting frequency, ending frequency, receptance in gap)
    '''
    Y=[]
    for config in configbandgaps:
        Yi=None
        for low,high,strength in config:
            if low<=frequency and frequency<=high:
                Yi=np.array([[strength]])
                break
        if Yi is None:
            Yi=np.array([[nominal_receptance]])
        Y.append(Yi)
    return Y

nominal_receptance=0.95
config1=[(100,1000,.5)]
config2=[(2000,10000,.5)]
frequencies=[coeff*10**count for count in range(6) for coeff in [.5,1]]
recipe=[0,1]

Yis=[get_block_receptances_at_frequency(f,nominal_receptance,[config1,config2]) for f in frequencies]
thomas_start=default_timer()
usthomas=np.array([block_thomas.compute_overall_frf(Yis[i],recipe,np.ones(2)) for i in range(len(Yis))])
thomas_end=default_timer()
usconventional=np.array([conventional.compute_overall_frf(Yis[i],recipe,np.ones(2)) for i in range(len(Yis))])
conventional_end=default_timer()
print(f"{len(frequencies)} frequencies. Thomas={thomas_end-thomas_start:.2f}, Conventional={conventional_end-thomas_end:.2f}")
fig=pyplot.figure()
ax=pyplot.axes()
for i in range(len(recipe)):
    ax.semilogx(frequencies,usthomas[:,i],":o",label=f"block Thomas [{i}]")
    ax.semilogx(frequencies,usconventional[:,i],":x",label=f"Conventional [{i}]")
ax.set_xlabel("Frequency")
ax.set_ylabel("Gain")
ax.legend()
pyplot.show()
