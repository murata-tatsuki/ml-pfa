from dataset import ILCDataset
import tools.load_awkward as la
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Specify input file (required)')
    parser.add_argument('-o', '--output', type=str, required=True, help='Specify output file (required)')
    parser.add_argument('--maximumTime', type=int, default=14)    
    parser.add_argument('--minimumPt', type=float, default=0.3)
    parser.add_argument('--nstart', type=int, default=0)    
    parser.add_argument('--nend', type=int, default=-1)    
    args = parser.parse_args()

    ak_feats, ak_labels = la.load_awkward2(args.input)
    ak_feats, ak_labels = ILCDataset.timingCut(ak_feats, ak_labels, args.maximumTime,args.minimumPt,args.nstart,args.nend)

    print(f'Saving to {args.output}')
    la.save_awkward(args.output, ak_feats, ak_labels)
    print('done')

if __name__ == '__main__':
    pass
    main()
    # debug()
    # run_profile()
