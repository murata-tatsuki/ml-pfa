import load_awkward as la
import numpy as np
import awkward as ak
import sys
import glob
import os

inputdir=sys.argv[1]
outdir=sys.argv[2]

print("input dir:", inputdir, ", output dir:", outdir)

files = list(sorted(glob.iglob(inputdir + '/*.h5')))

for path in files:
    print("processing ", path)
    ak_feats, ak_labels = la.load_awkward2(path)
    basename=os.path.splitext(os.path.basename(path))[0]

    n = ak.num(ak_feats, axis=0)

    for i in range(n):
        feat = ak_feats[i].to_numpy()
        label = ak_labels[i].to_numpy()
        outfile = outdir + '/' + basename + '_' + str(i) + '.npz'

        np.savez(outfile, recHitFeatures=feat, recHitTruthClusterIdx=label)

