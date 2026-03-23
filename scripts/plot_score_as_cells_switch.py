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
config1=[]#config 1 has no bandgap
config2=[(2000,10000,np.diag(n*[0.5]))]#config 2 has a notch containing the frequency of interest
target_frequency=5000
Yis=get_block_receptances_at_frequency(target_frequency,nominal_receptance,[config1,config2])
N=72
excitation=np.array([1 for _ in range(n)]+[0 for _ in range(n*(N-1))])#unit excitation on each DOF of first cell, zero elsewhere

thomas_start=default_timer()
ufromstart=[]
ufromend=[]
uspaced=[]
for num_switched in range(N):
    #switch from the leading edge
    recipe=[1]*num_switched+[0]*(N-num_switched)
    ufromstart.append(block_thomas.compute_overall_frf(Yis,recipe,excitation))

    #switch from the trailing edge
    recipe=[0]*(N-num_switched)+[1]*num_switched
    ufromend.append(block_thomas.compute_overall_frf(Yis,recipe,excitation))

    #space the switches out evenly
    recipe=[0]*N
    if num_switched%2==0:
        #even so 2 go at the ends
        n_gaps=num_switched-1
        stride=(N-1)//n_gaps
        for j in range(num_switched):
            recipe[j*stride]=1
    else:
        #odd
        n_gaps=num_switched+1
        stride=(N-1)//n_gaps
        for j in range(num_switched):
            recipe[(j+1)*stride]=1
    uspaced.append(block_thomas.compute_overall_frf(Yis,recipe,excitation))
thomas_end=default_timer()
print(f"{N} unit cells, {n} cell DOF Thomas={thomas_end-thomas_start:.4f}s")

ufromstart=np.array(ufromstart)
ufromend=np.array(ufromend)
uspaced=np.array(uspaced)
fig=pyplot.figure()
ax=pyplot.axes()
ax.plot(range(N),ufromstart[:,-1],":o",label=f"Switch from Start")
ax.plot(range(N),ufromend[:,-1],":o",label=f"Switch from End")
ax.plot(range(N),uspaced[:,-1],":o",label=f"Space Switches Evenly")

ax.set_xlabel("Number of Cells in Bandgap Config")
ax.set_ylabel("Gain")
ax.legend()
pyplot.show()
