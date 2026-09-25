"""Compare new ROOT spectra/metrics against the independently built upstream binary."""
from pathlib import Path
from array import array
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import ROOT
from save_root_pandora_eval import Tree

UPSTREAM=Path(os.environ.get('LCPANDORA_ANALYSIS', '/home/murata/LCPandoraAnalysis'))

class ReferenceIntegration(unittest.TestCase):
    @unittest.skipUnless((UPSTREAM/'performance/AnalysePerformance.cc').exists() and
                         shutil.which('g++') and shutil.which('root-config'),
                         'LCPandoraAnalysis sources/compiler unavailable')
    def test_upstream_histograms_resolution_and_event_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            binary=work/'upstream'
            flags=shlex.split(subprocess.check_output(['root-config','--cflags','--libs'],text=True))
            subprocess.run(['g++','-O2',str(UPSTREAM/'performance/AnalysePerformance.cc'),
                            str(UPSTREAM/'src/AnalysisHelper.cc'),'-I'+str(UPSTREAM/'include'),
                            *flags,'-o',str(binary)],check=True)
            out=str(work/'fixture.root')
            f=ROOT.TFile(out,'RECREATE')
            e=Tree('eval_events',integers=('event','qPdg','reference_valid','common_valid','three_way_valid','truth_valid','truth_complete','source_index','inference_requested'),floats=('pfoEnergyTotal','mcEnergyENu','thrust'),doubles=('gnn_energy','truth_energy'),strings=('source_id',))
            r=Tree('PfoAnalysisTree',floats=('pfoEnergyTotal','mcEnergyENu','thrust'))
            q=array('i',[1]);r.tree.Branch('qPdg',q,'qPdg/I')
            m=Tree('eval_matches',integers=('event','algorithm','matched'),doubles=('efficiency','purity','reco_energy','truth_energy'))
            rng=np.random.RandomState(13)
            for i in range(180):
             p=np.float32(350+rng.normal(0,15)+(60 if i%19==0 else 0))
             nu=np.float32(2*(i%7));theta=np.float32(.25 if i<90 else .85)
             vals=dict(event=i,qPdg=1,reference_valid=1,common_valid=int(i%11!=0),three_way_valid=int(i%11!=0 and i%3!=0),truth_valid=1,truth_complete=int(i%3!=0),source_index=i,inference_requested=1,pfoEnergyTotal=float(p),mcEnergyENu=float(nu),thrust=float(theta),gnn_energy=float(p*1.03),truth_energy=float(p*.99),source_id='fixture')
             e.fill(vals);r.fill(vals)
            ROOT.TObjString('{}').Write('pandora_eval_metadata')
            ROOT.TObjString('fixture-only').Write('comparison_configuration')
            f.Write();f.Close()
            macro=Path(__file__).resolve().parent/'macro/src/efficiency_purity_check_reco_effpur_contiribution_pandora_eval.cxx'
            ours=str(work/'performance.root')
            oracle=str(work/'upstream.root')
            subprocess.run(['root','-l','-b','-q',f'{macro}("{out}","{ours}")'],check=True)
            subprocess.run([str(binary),out,oracle],check=True)
            a=ROOT.TFile(ours);b=ROOT.TFile(oracle)
            for name,ref in [('pandora_reference_central','fPFA_L7A')]+[(f'pandora_reference_angle_{i}',f'fPFA_{i}') for i in range(13)]:
             ah=a.Get(name);bh=b.Get(ref)
             assert ah.GetEntries()==bh.GetEntries(),name
             assert all(ah.GetBinContent(i)==bh.GetBinContent(i) for i in range(ah.GetNcells())),name
            res=b.Get('ResVsCosTheta')
            for row in a.Get('resolution'):
             if row.sample==0 and row.angle_bin>=0 and row.valid:
              assert row.resolution_percent==res.GetBinContent(row.angle_bin+1),(row.resolution_percent,res.GetBinContent(row.angle_bin+1))
              assert row.error_percent==res.GetBinError(row.angle_bin+1),(row.error_percent,res.GetBinError(row.angle_bin+1))
            assert a.Get('pandora_common_central').GetEntries()==a.Get('gnn_common_central').GetEntries()
            assert a.Get('pandora_three_way_central').GetEntries()==a.Get('gnn_three_way_central').GetEntries()==a.Get('truth_three_way_central').GetEntries()
            print('PASS: all 14 upstream spectra match bin-for-bin; populated angular resolution/errors exactly match; common and complete-truth sets agree')
            a.Close();b.Close()

if __name__=='__main__': unittest.main()
