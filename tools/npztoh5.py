import load_awkward as la
import numpy as np
import awkward as ak
import sys

file=sys.argv[1]
outfile=sys.argv[2]

print("input file list:", file, ", output file:", outfile)

with open(file, "r") as f:
    npzs = f.read().split()

a = [np.load(f) for f in npzs]
feats = [aa['recHitFeatures'] for aa in a]
labels = [aa['recHitTruthClusterIdx'] for aa in a]
ak_feats_list = [ak.from_numpy(f) for f in feats]
ak_labels_list = [ak.from_numpy(l) for l in labels]
ak_feats = ak.Array(ak_feats_list)
ak_labels = ak.Array(ak_labels_list)

la.save_awkward(outfile,ak_feats,ak_labels)
