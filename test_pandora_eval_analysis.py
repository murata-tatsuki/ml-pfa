"""Read-only sample checks and synthetic edge cases for the new analysis path."""
import ast
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch_geometric.data import Data
from dataset import ILCDataset, incremental_cluster_index_np, checked_int64_ids
from pandora_eval_data import iter_events, model_data, validate_event, gap_hit_mask
from pandora_eval_reconstruction import pandora_clusters, overlap_metrics, energy_clusters, configure_regression_output

SAMPLES = Path('/home/murata/simulation/validation/pandora_eval/final')

def synthetic():
    feat=np.zeros((3,13)); feat[:,1:4]=[[1,2,3],[2,3,4],[3,4,5]]
    feat[:,0]=[1.,2.,0.]; feat[2,5]=1; feat[2,7]=3.
    label=np.zeros((3,9)); label[:,1]=[0,-1,-1]; label[0,0]=101
    row=np.array([[0,0,0,1,0,1,1,1,1,101,10],
                  [0,0,1,1,-1,0,1,1,0,102,-1],
                  [1,1,0,7,-1,0,1,1,1,-103,-1]],dtype=float)
    pand=np.zeros((3,18)); pand[:,9]=row[:,9]
    pfo=np.array([[0,501,4.,211,1,2,3,1,0],[1,502,8.,22,1,2,3,0,1]],dtype=float)
    ev=np.zeros(23); ev[0]=12;ev[1]=2;ev[5:8]=1;ev[8]=-1;ev[17]=12
    return dict(feature=feat,label=label,row_info=row,pandora=pand,pfo=pfo,
        pfo_links=np.array([[0,1,103,2,-1],[1,0,101,0,601],[1,0,102,1,601]],dtype=float),
        truth_particles=np.array([[0,10,22,9.,0,0,9.]]),cluster=np.array([[0,0.]]),
        event=np.zeros(10),event_eval=ev,global_index=0,collections=['calo','tracks'])

class ReaderTests(unittest.TestCase):
    def test_linear_checkpoint_compatibility_preserves_weights_and_clustering_head(self):
        model=torch.nn.Module()
        model.head_specs=[dict(kind='clustering'),dict(kind='regression')]
        model.head_output=torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2,1)),
            torch.nn.Sequential(torch.nn.Linear(2,1),torch.nn.Softplus())])
        with torch.no_grad():
            model.head_output[1][0].weight.zero_();model.head_output[1][0].bias.zero_()
        weights={k:v.clone() for k,v in model.state_dict().items()}
        cluster=model.head_output[0]
        configure_regression_output(model,'current')
        self.assertAlmostEqual(model.head_output[1](torch.zeros(1,2)).item(),np.log(2),places=6)
        configure_regression_output(model,'linear')
        self.assertEqual(model.head_output[1](torch.zeros(1,2)).item(),0.)
        self.assertIs(model.head_output[0],cluster)
        for k,v in weights.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        model.cluster_energy_pooling=True
        with self.assertRaises(ValueError):configure_regression_output(model,'linear')

    def test_gap_mask_is_before_forward_and_preserves_original_row_identity(self):
        e=synthetic()
        e['collections'].append('EcalBarrelCollectionGapHits')
        e['row_info'][0,1]=2  # Even a truth-labelled gap hit must be excluded.
        before=copy.deepcopy(e)
        d=model_data(e,exclude_gap_hits=True)
        self.assertEqual(set(d.input_row.tolist()),{1,2})
        self.assertEqual(int((d.x[:,4]>.5).sum()),1)
        self.assertEqual(int((~d.truth_valid).sum()),2) # Non-gap unknowns survive.
        self.assertEqual(sum(c['energy'] for c in pandora_clusters(e)),12.)
        for pos,row in enumerate(d.input_row.tolist()):
            np.testing.assert_array_equal(d.feat[pos].numpy(),e['feature'][row].astype(np.float32))
            self.assertEqual(int(d.hitid[pos]),int(e['row_info'][row,9]))
        for key in ('feature','label','row_info','pandora','pfo_links'):
            np.testing.assert_array_equal(e[key],before[key])
        e['collections'][2]='EcalEndcapsCollectionGapHits'
        self.assertEqual(set(model_data(e,exclude_gap_hits=True).input_row.tolist()),{1,2})
        e['feature'][1,1]=np.nan
        self.assertEqual(model_data(e,exclude_gap_hits=True).input_row.tolist(),[2])

    def test_unknown_hits_and_tracks_survive_with_exact_ids(self):
        e=synthetic(); validate_event(e); d=model_data(e)
        self.assertEqual(len(d.x),3)
        self.assertEqual(int((d.y[:,0]==0).sum()),2)
        self.assertEqual(int(d.y[d.input_row==2,1]),1)
        self.assertEqual(int(d.hitid[d.input_row==2]),-103)
        np.testing.assert_array_equal(d.row_info[:,9].numpy(),d.hitid.numpy())
        for pos,row in enumerate(d.input_row):
            np.testing.assert_array_equal(d.feat[pos].numpy(),e['feature'][int(row)].astype(np.float32))

    def test_truth_changes_do_not_select_or_change_model_features(self):
        e=synthetic(); d=model_data(e)
        e['label'][1,1]=0; e['row_info'][1,5]=1
        other=model_data(e)
        np.testing.assert_array_equal(d.x[d.input_row.argsort()].numpy(),other.x[other.input_row.argsort()].numpy())

    def test_invalid_inputs_do_not_change_pfo_sum(self):
        e=synthetic(); e['feature'][2,1]=np.nan;e['row_info'][2,6]=0
        d=model_data(e)
        self.assertEqual(len(d.x),2)
        self.assertEqual(sum(c['energy'] for c in pandora_clusters(e)),12)
        self.assertEqual(pandora_clusters(e)[0]['energy'],4) # track-only PFO

    def test_large_integer_identity(self):
        e=synthetic(); obj=2**24+17
        e['row_info'][2,9]=-obj; e['pandora'][2,9]=-obj;e['pfo_links'][0,2]=obj
        d=model_data(e)
        self.assertEqual(int(d.hitid[d.input_row==2]),-obj)

    def test_malformed_lengths_and_link_fail(self):
        e=synthetic(); e['label']=e['label'][:-1]
        with self.assertRaises(ValueError):validate_event(e)
        e=synthetic();e['pfo_links'][0,3]=99
        with self.assertRaises(ValueError):validate_event(e)

    def test_empty_and_singleton(self):
        e=synthetic()
        for k in ('feature','label','row_info','pandora'):e[k]=e[k][:0]
        d=model_data(e)
        self.assertEqual(tuple(d.x.shape),(0,11))
        e=synthetic()
        for k in ('feature','label','row_info','pandora'):e[k]=e[k][2:]
        d=model_data(e)
        self.assertEqual(tuple(d.x.shape),(1,11))
        self.assertEqual(d.y.tolist(),[[0,1]])

    def test_labelled_overlap_excludes_unknown_denominators(self):
        e=synthetic();metrics=list(overlap_metrics(e,pandora_clusters(e),[0,1,2]))
        self.assertEqual(len(metrics),1)
        self.assertEqual(metrics[0]['efficiency'],1)
        self.assertEqual(metrics[0]['purity'],1)
        self.assertEqual(metrics[0]['reco_energy'],8)
        self.assertEqual(metrics[0]['reco_known_deposit'],1)
        unmatched=list(overlap_metrics(e,[],[0,1,2]))[0]
        self.assertEqual(unmatched['efficiency'],0)
        self.assertFalse(unmatched['matched'])

    def test_truth_noise_never_forms_a_cluster(self):
        d=model_data(synthetic());pred=dict(beta=np.array([.2,.3,.9]),tracker=np.ones(3),calo=np.ones(3),assignments=d.y[:,0].numpy())
        cs=energy_clusters(d,pred)
        self.assertEqual(len(cs),1)
        self.assertEqual(cs[0]['members'],{0})

    def test_energy_policy_is_explicit(self):
        d=model_data(synthetic())
        beta=np.where(d.input_row.numpy()==0,.99,.1)
        pred=dict(beta=beta,tracker=np.where(d.x[:,4].numpy()>.5,7.,99.),calo=np.array([1.,2.,3.]),assignments=np.ones(3,dtype=int))
        self.assertEqual(energy_clusters(d,pred,'alpha')[0]['energy'],6.)
        self.assertEqual(energy_clusters(d,pred,'any-track')[0]['energy'],7.)

    def test_legacy_preprocessing_is_unchanged(self):
        # Execute only HEAD's original method, keeping it independent of our edits.
        original=subprocess.check_output(['git','show','HEAD:dataset.py'],text=True)
        tree=ast.parse(original)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ILCDataset')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='featurize_from_numpy')
        fn.decorator_list=[]
        ns=dict(np=np,torch=torch,Data=Data,incremental_cluster_index_np=incremental_cluster_index_np,checked_int64_ids=checked_int64_ids)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original>','exec'),ns)
        e=synthetic();e['label'][:,1]=[0,1,1];e['label'][:,0]=[101,102,-103]
        ds=SimpleNamespace(mctpe=False,thetaphi=True,momentum=True,momentumAmp=True,max_momentum=3.,
            test_mode=True,pandora=True,event_energy=False,noise_index=-1,
            shaper_tanh=lambda x,a,b,c,d:a*np.tanh(b*(x-c))+d)
        args=(e['feature'],e['label'],e['pandora'],None,None,0,ds)
        old=ns['featurize_from_numpy'](*copy.deepcopy(args))
        new=ILCDataset.featurize_from_numpy(*copy.deepcopy(args))
        for key in old.keys:
            torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)

    @unittest.skipUnless(SAMPLES.exists(),'KEK sample H5s unavailable')
    def test_all_samples_counts_and_identity(self):
        events=rows=unknown=pfos=0
        for energy in (40,91,200,350,500):
            for e in iter_events(str(SAMPLES/f'dd_{energy}_sample.h5')):
                d=model_data(e);events+=1;rows+=len(d.x);unknown+=int((~d.truth_valid).sum());pfos+=len(e['pfo'])
                self.assertEqual(len(d.x),len(e['feature']))
                self.assertAlmostEqual(sum(c['energy'] for c in pandora_clusters(e)),e['event_eval'][0])
        self.assertEqual((events,rows,unknown,pfos),(16,77502,1203,872))

    def test_no_legacy_index_does_not_filter(self):
        path=SAMPLES.parent/'no_legacy_20260925/dd_350_sample.h5'
        if not path.exists():self.skipTest('Sample unavailable')
        events=list(iter_events(str(path)))
        self.assertEqual(len(events),2)
        self.assertTrue(all(e['event_eval'][8]==-1 for e in events))
        self.assertTrue(all(len(model_data(e).x)>0 for e in events))

if __name__=='__main__':unittest.main()
