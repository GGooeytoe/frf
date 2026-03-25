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

n=3#degrees of freedom in each unit cell
nominal_receptance=np.diag([0.75,0.85,0.95])
config1=[(1000,3000,np.diag(n*[0.5]))]#config 1 has a notch around one frequency
config2=[(4500,5500,np.diag(n*[0.5]))]#config 2 has a notch containing the other frequency
target_frequencies=[2000,5000]
Yis_by_freq=[get_block_receptances_at_frequency(f,nominal_receptance,[config1,config2]) for f in target_frequencies]
N=72
excitation=np.array([1 for _ in range(n)]+[0 for _ in range(n*(N-1))])#unit excitation on each DOF of first cell, zero elsewhere

thomas_start=default_timer()
ufromstart=[]
ufromend=[]
uspaced=[]
for i,freq in enumerate(target_frequencies):
    Yis=Yis_by_freq[i]
    ufromstarti=[]
    ufromendi=[]
    uspacedi=[]
    for num_switched in range(N):
        #switch from the leading edge
        recipe=[1]*num_switched+[0]*(N-num_switched)
        ufromstarti.append(block_thomas.compute_overall_frf(Yis,recipe,excitation))

        #switch from the trailing edge
        recipe=[0]*(N-num_switched)+[1]*num_switched
        ufromendi.append(block_thomas.compute_overall_frf(Yis,recipe,excitation))

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
            if stride==0:
                stride=1
            for j in range(num_switched):
                recipe[(j+1)*stride]=1
        uspacedi.append(block_thomas.compute_overall_frf(Yis,recipe,excitation))
    ufromstart.append(ufromstarti)
    ufromend.append(ufromendi)
    uspaced.append(uspacedi)
thomas_end=default_timer()
print(f"{N} unit cells, {n} cell DOF, {len(target_frequencies)} frequencies, Thomas={thomas_end-thomas_start:.4f}s")

ufromstart=np.array(ufromstart)
ufromend=np.array(ufromend)
uspaced=np.array(uspaced)
fig=pyplot.figure()
axes=fig.subplots(len(target_frequencies),n,sharex=True,sharey=True)
for i,freq in enumerate(target_frequencies):
    for j in range(n):
        axes[i,j].plot(range(N),ufromstart[i,:,j],":o",label=f"Switch from Start")
        axes[i,j].plot(range(N),ufromend[i,:,j],":o",label=f"Switch from End")
        axes[i,j].plot(range(N),uspaced[i,:,j],":o",label=f"Space Switches Evenly")
    axes[i,0].set_ylabel(f"Gain (Freq={freq})")
for j in range(n):
    axes[0,j].set_title(f"DOF={j}")
    axes[-1,j].set_xlabel("Number of Cells in Config 2")
axes[-1,-1].legend()
pyplot.show()
