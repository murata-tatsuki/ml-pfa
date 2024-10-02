import os
import sys
import numpy as np
import awkward as ak
from colorwheel import ColorWheel
import tools.load_awkward # local loader
import matplotlib.pyplot as plt



def plot_stats(stats_numpy,hist_min,hist_max,nbins=50,key_name="plot"):
    fig = plt.figure(facecolor="white",figsize=[6,4])
    ax = fig.gca()
    bins = np.linspace(hist_min, hist_max, nbins)
    ax.hist(stats_numpy, bins=bins, linewidth=2)    
    #average_hist = np.round(np.nanmean(stats_numpy),decimals=5)
    #ax.text(0.85,0.85,f"average : {average_hist}")
    plt.show()
    fig.savefig(f"{key_name}.png")


def main():
    if(len(sys.argv) != 2):
        print("Usage: python display_h5_energy.py inputfile")
        sys.exit()

    awkfile=sys.argv[1]
    print("inputfile:", awkfile)

    ak_feat, ak_label, ak_pred, ak_energy = tools.load_awkward.load_awkward2(awkfile)

    eA = ak.to_numpy( ak_energy[:,0] )
    eB = ak.to_numpy( ak_energy[:,1] )
    eC = ak.to_numpy( ak_energy[:,2] )
    eD = ak.to_numpy( ak_energy[:,3] )
    
    # print(f"{ak_energy[0]=}")
    # print(f"{eA=}")
    # print(f"{eB=}")
    # print(f"{eC=}")
    # print(f"{eD=}")
    # print(f"{len(eA)=}")

    #print(np.count_nonzero(np.isnan(eA) + np.isnan(eB)))
    #print(np.count_nonzero(np.isnan(eC) + np.isnan(eD)))

    select = ((eA+eB)*(eC+eD)>0) & ~ np.isinf(eA+eB)

    eA = eA[ select ]
    eB = eB[ select ]
    eC = eC[ select ]
    eD = eD[ select ]

    eff_charged = eA/(eA+eB)
    #pur_charged = eA/(eA+eC)
    eff_neutral = eD/(eC+eD)
    #pur_neutral = eD/(eB+eD)

    basename = os.path.splitext(awkfile)[0]
    
    plot_stats(eff_charged, 0, 1, 100, f"{basename}_eff_charged")
    plot_stats(eff_neutral, 0, 1, 100, f"{basename}_eff_neutral")

    print(f"{np.average(eff_charged)=}, {np.median(eff_charged)=}")
    print(f"{np.average(eff_neutral)=}, {np.median(eff_neutral)=}")

if __name__=='__main__':
    main()
