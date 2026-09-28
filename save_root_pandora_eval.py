#!/usr/bin/env python3
"""Write aligned Pandora/GNN/truth diagnostics from pandora-eval-1 H5.

Outputs a new schema; use the copied *_pandora_eval.cxx macro, not the legacy
five-tree macro. Existing checkpoints are used without training.
"""
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import ROOT
import torch
from model import get_model
from cluster_energy import pooling_model
from pandora_eval_data import iter_events, model_data, gap_hit_mask
from pandora_eval_reconstruction import predict, energy_clusters, pandora_clusters, overlap_metrics, configure_regression_output

SCHEMA = 'pandora-comparison-root-1'

class Tree:
    def __init__(self, name, integers=(), doubles=(), floats=(), strings=()):
        self.tree = ROOT.TTree(name, name)
        self.buffers = {}
        for fields, typ, leaf in [(integers, 'q', 'L'), (doubles, 'd', 'D'), (floats, 'f', 'F')]:
            for field in fields:
                self.buffers[field] = array(typ, [0])
                self.tree.Branch(field, self.buffers[field], f'{field}/{leaf}')
        for field in strings:
            self.buffers[field] = ROOT.std.string()
            self.tree.Branch(field, self.buffers[field])
        self.strings = set(strings)

    def fill(self, values):
        for k, b in self.buffers.items():
            if k not in values:
                raise ValueError(f'Missing {self.tree.GetName()}.{k}')
            if k in self.strings:
                b.assign(str(values[k]))
            else:
                b[0] = values[k]
        self.tree.Fill()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(args, model=None):
    if not 0 <= args.tbeta <= 1 or not np.isfinite(args.td) or args.td <= 0:
        raise ValueError('Require 0 <= tbeta <= 1 and finite td > 0')
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(f'{output}: choose a new output or pass --replace')
    if not args.pandora_only and not args.checkpoint:
        raise ValueError('--checkpoint is required unless --pandora-only is selected')
    torch.set_num_threads(args.threads)
    if model is None and not args.pandora_only:
        model = get_model(args.checkpoint, jit=False,
            input_dim=args.input_dim + (3 + int(args.momentum_amp) if args.momentum else 0),
            output_dim=args.output_dim + 1 + int(args.calo_head),
            energy_regression=True, energy_regression_cluster=args.calo_head,
            model_variant=args.model_variant, cluster_energy_source='predicted',
            cluster_energy_tbeta=args.tbeta, cluster_energy_td=args.td).to(args.device).eval()
        configure_regression_output(model, args.regression_output_activation)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp.root', dir=output.parent)
    os.close(fd)
    file = ROOT.TFile(staging, 'RECREATE')
    if file.IsZombie():
        raise OSError(f'Cannot create {staging}')
    try:
        events = Tree('eval_events',
            integers=('event', 'source_index', 'run_number', 'event_number', 'h5_index', 'legacy_index',
                      'legacy_columns_checked', 'qPdg', 'n_inputs', 'n_model_inputs', 'n_unknown_truth',
                      'n_invalid_inputs', 'n_pfos', 'gnn_valid', 'truth_valid', 'truth_complete',
                      'reference_valid', 'common_valid', 'three_way_valid', 'n_unassigned_gnn',
                      'legacy_validation_available', 'inference_requested', 'n_excluded_gap_hits',
                      'n_model_hits', 'n_model_tracks'),
            doubles=('pandora_energy', 'mc_energy_enu', 'gnn_energy', 'truth_energy',
                     'gnn_energy_alpha', 'gnn_energy_any_track', 'truth_energy_alpha',
                     'truth_energy_any_track', 'unknown_truth_deposit'),
            floats=('pfoEnergyTotal', 'mcEnergyENu', 'thrust', 'pandora_corrected', 'gnn_corrected', 'truth_corrected'),
            strings=('source_id', 'source_path', 'input_path', 'collections_json', 'inference_error'))
        # Drop-in reference tree for the unmodified LCPandoraAnalysis performance binary.
        reference = Tree('PfoAnalysisTree', integers=(), doubles=(),
                         floats=('pfoEnergyTotal', 'mcEnergyENu', 'thrust'))
        # Upstream expects qPdg/I, not Long64_t.
        ref_pdg = array('i', [0]); reference.tree.Branch('qPdg', ref_pdg, 'qPdg/I')
        pfos = Tree('pfo', integers=('event','pfo_index','object_id','type','n_tracks','n_clusters'),
                    doubles=('energy','px','py','pz'))
        links = Tree('pfo_links', integers=('event','pfo_index','kind','object_id','input_row','cluster_id'))
        hits = Tree('eval_hits', integers=('event','input_row','input_rank','kind','collection','element',
                    'detector','legacy_row','truth_valid','feature_valid','model_input','object_id',
                    'truth_label','gnn_cluster','truth_cluster'), doubles=('deposit','beta','tracker_energy','calo_energy'))
        clusters = Tree('eval_clusters', integers=('event','algorithm','cluster','n_tracks','n_inputs','seed_input_row'),
                        doubles=('energy','energy_alpha','energy_any_track'))
        matches = Tree('eval_matches', integers=('event','algorithm','truth_label','truth_pdg','matched','cluster',
                        'truth_n_inputs','overlap_n_inputs'), doubles=('truth_energy','reco_energy','overlap_energy',
                        'truth_deposit','reco_known_deposit','efficiency','purity'))
        truths = Tree('eval_truth', integers=('event','truth_label','object_id','pdg'), doubles=('energy','px','py','pz'))
        all_trees = [events,reference,pfos,links,hits,clusters,matches,truths]
        provenance, count, failures = {}, 0, 0
        for event in iter_events(args.input, args.start, args.stop):
            i = count; count += 1
            data = model_data(event, args.input_dim, args.momentum, args.momentum_amp, args.exclude_gap_hits)
            selected = data.input_row.numpy()
            n = len(event['feature']); ev = event['event_eval']
            n_gap = int(gap_hit_mask(event).sum()) if args.exclude_gap_hits else 0
            unknown = ~data.truth_valid.numpy()
            input_rank = np.full(n, -1, dtype=np.int64); input_rank[selected] = np.arange(len(selected))
            gnn_assignment = np.full(n, -1, dtype=np.int64)
            truth_assignment = np.full(n, -1, dtype=np.int64)
            # 0 is unassigned/unknown; no unknown MC cluster is created.
            truth_assignment[selected] = data.y[:, 0].numpy()
            beta = np.full(n, np.nan); tracker = beta.copy(); calo = beta.copy()
            predictions = {}; gnn_ok = truth_ok = False; error = ''
            if model is not None:
                try:
                    if len(selected):
                        prediction = predict(model, data, args.device, args.tbeta, args.td, args.calo_head)
                        predictions[1] = energy_clusters(data, prediction, args.energy_policy)
                        gnn_assignment[selected] = prediction['assignments']
                        beta[selected] = prediction['beta']; tracker[selected] = prediction['tracker']
                        if prediction['calo'] is not None:
                            calo[selected] = prediction['calo']
                        gnn_ok = True
                        if pooling_model(model) is not None:
                            truth_prediction = predict(model, data, args.device, args.tbeta, args.td, args.calo_head, truth=True)
                        else:
                            truth_prediction = dict(prediction, assignments=data.y[:, 0].numpy())
                        predictions[2] = energy_clusters(data, truth_prediction, args.energy_policy)
                        truth_ok = True
                    elif n == n_gap:
                        predictions[1] = []; predictions[2] = []
                        gnn_ok = truth_ok = True
                    else:
                        error = 'No valid model inputs'
                except (RuntimeError, ValueError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    if not args.keep_failed_events:
                        raise RuntimeError(f"{event['source']['id']}:{int(ev[9])}: {error}") from exc
                if not (gnn_ok and truth_ok):
                    failures += 1
            predictions[0] = pandora_clusters(event)
            def total(algorithm, name='energy'):
                return sum(c[name] for c in predictions[algorithm]) if algorithm in predictions else float('nan')
            truth_complete = bool(len(selected) == n and not np.any(unknown))
            reference_valid = bool(ev[5] and ev[6] and ev[7] and np.isfinite(ev[[0,2,3,17,18]]).all())
            common = bool(reference_valid and gnn_ok)
            provenance[event['input_path']] = dict(metadata=event['metadata'], collections=event['collections'])
            values = dict(event=i, source_index=int(ev[9]), run_number=int(ev[10]), event_number=int(ev[11]),
                h5_index=int(ev[19]), legacy_index=int(ev[8]), legacy_columns_checked=int(ev[21]),
                legacy_validation_available=int(event['metadata'].get('legacy_validation_available',
                                                                     bool(event['metadata'].get('legacy_baseline')))),
                inference_requested=int(model is not None), qPdg=int(ev[4]), n_inputs=n, n_model_inputs=len(selected), n_unknown_truth=int(unknown.sum()),
                n_invalid_inputs=n-n_gap-len(selected), n_excluded_gap_hits=n_gap,
                n_model_tracks=int((data.row_info[:,0] == 1).sum()),
                n_model_hits=int((data.row_info[:,0] == 0).sum()), n_pfos=len(event['pfo']), gnn_valid=int(gnn_ok),
                truth_valid=int(truth_ok), truth_complete=int(truth_complete), reference_valid=int(reference_valid),
                common_valid=int(common), three_way_valid=int(common and truth_ok and truth_complete),
                n_unassigned_gnn=int(np.count_nonzero(gnn_assignment[selected] <= 0)) if gnn_ok else -1,
                pandora_energy=float(ev[0]), mc_energy_enu=float(ev[2]),
                gnn_energy=total(1), truth_energy=total(2), gnn_energy_alpha=total(1,'energy_alpha'),
                gnn_energy_any_track=total(1,'energy_any_track'), truth_energy_alpha=total(2,'energy_alpha'),
                truth_energy_any_track=total(2,'energy_any_track'),
                unknown_truth_deposit=float(event['feature'][selected[unknown],0].sum()),
                pfoEnergyTotal=float(ev[17]), mcEnergyENu=float(ev[18]), thrust=float(ev[3]),
                pandora_corrected=float(np.float32(np.float32(ev[17])+np.float32(ev[18]))),
                gnn_corrected=float(np.float32(np.float32(total(1))+np.float32(ev[18]))),
                truth_corrected=float(np.float32(np.float32(total(2))+np.float32(ev[18]))),
                source_id=event['source']['id'], source_path=event['source']['path'], input_path=event['input_path'],
                collections_json=json.dumps(event['collections']), inference_error=error)
            events.fill(values)
            if reference_valid:
                ref_pdg[0] = int(ev[4]); reference.fill(values)
            if args.detail == 'full':
                for p in event['pfo']:
                    pfos.fill(dict(event=i, **dict(zip(('pfo_index','object_id','energy','type','px','py','pz','n_tracks','n_clusters'),
                        [int(p[0]),int(p[1]),float(p[2]),int(p[3]),*map(float,p[4:7]),int(p[7]),int(p[8])]))))
                for link in event['pfo_links']:
                    links.fill(dict(event=i, **dict(zip(('pfo_index','kind','object_id','input_row','cluster_id'),map(int,link)))))
                for t in event['truth_particles']:
                    truths.fill(dict(event=i, truth_label=int(t[0]),object_id=int(t[1]),pdg=int(t[2]),
                        energy=float(t[3]),px=float(t[4]),py=float(t[5]),pz=float(t[6])))
                for row, info in enumerate(event['row_info']):
                    hits.fill(dict(event=i,input_row=row,input_rank=int(input_rank[row]),kind=int(info[0]),collection=int(info[1]),
                        element=int(info[2]),detector=int(info[3]),legacy_row=int(info[4]),truth_valid=int(info[5]),
                        feature_valid=int(info[6]),model_input=int(input_rank[row]>=0),object_id=int(info[9]),
                        truth_label=int(event['label'][row,1]),gnn_cluster=int(gnn_assignment[row]),
                        truth_cluster=int(truth_assignment[row]),deposit=float(event['feature'][row,0]),
                        beta=float(beta[row]),tracker_energy=float(tracker[row]),calo_energy=float(calo[row])))
            if args.detail != 'events':
                for algorithm, cs in predictions.items():
                    for c in cs:
                        clusters.fill(dict(event=i,algorithm=algorithm,**c))
                    if args.detail == 'full':
                        for match in overlap_metrics(event, cs, selected):
                            matches.fill(dict(event=i,algorithm=algorithm,**match))
            print(f"event={i} source_index={int(ev[9])} inputs={n} used={len(selected)} "
                  f"unknown_truth={int(unknown.sum())} PFOs={len(event['pfo'])} gnn_valid={gnn_ok}", flush=True)
            for tree in all_trees:
                if count % 10 == 0:
                    tree.tree.FlushBaskets()
        here = Path(__file__).resolve().parent
        metadata = dict(schema=SCHEMA, arguments=vars(args), inputs=provenance, events=count,
            inference_failures=failures, algorithms={'0':'Pandora','1':'GNN','2':'truth-labelled-inputs'},
            energy_policy=args.energy_policy,
            truth_scope='Only labelled model-input rows; three_way_valid requires complete input truth coverage',
            match_scope='Best deposited-overlap per truth, not bijective; denominators use labelled model inputs',
            input_scope=('Finite feature-valid candidates excluding the two ECAL GapHits collections before forward'
                         if args.exclude_gap_hits else 'All finite feature-valid stored candidates')
                        + '; no truth, timing, pT or PFO membership cut',
            detail=args.detail, exclude_gap_hits=args.exclude_gap_hits,
            exact_pandora_internal_selection=False,
            checkpoint_sha256=sha256(args.checkpoint) if model is not None else None,
            code_sha256={name:sha256(here/name) for name in ('dataset.py','extended_h5.py','pandora_eval_data.py',
                'pandora_eval_reconstruction.py','save_root_pandora_eval.py', 'model.py', 'gravnet_model.py')})
        file.cd()
        # Comparable files must share these settings, but may have different
        # source paths, event ranges and H5 creation times.
        configurations = [dict(analysis_settings=x['metadata']['analysis_settings'],
                               input_inventory=x['metadata']['input_inventory']) for x in provenance.values()]
        configuration = dict(input=configurations[0] if configurations else None,
            checkpoint_sha256=metadata['checkpoint_sha256'], energy_policy=args.energy_policy,
            input_dim=args.input_dim, output_dim=args.output_dim, momentum=args.momentum,
            momentum_amp=args.momentum_amp, calo_head=args.calo_head, tbeta=args.tbeta, td=args.td,
            regression_output_activation=args.regression_output_activation,
            model_variant=args.model_variant, exclude_gap_hits=args.exclude_gap_hits, detail=args.detail, code_sha256=metadata['code_sha256'])
        ROOT.TObjString(json.dumps(configuration, sort_keys=True)).Write('comparison_configuration')
        ROOT.TObjString(json.dumps(metadata,sort_keys=True)).Write('pandora_eval_metadata')
        file.Write(); file.Close()
        check = ROOT.TFile.Open(staging)
        if not check or check.IsZombie() or check.Get('eval_events').GetEntries() != count:
            raise RuntimeError('Output ROOT validation failed')
        check.Close()
        if args.replace:
            os.replace(staging,output)
        else:
            os.link(staging,output); os.unlink(staging)
        print(f'Wrote {count} events to {output}; inference failures={failures}')
    finally:
        if file.IsOpen():
            file.Close()
        if os.path.exists(staging):
            os.unlink(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='New H5 file, directory, or quoted glob')
    parser.add_argument('output', help='New comparison ROOT file')
    parser.add_argument('--checkpoint')
    parser.add_argument('--pandora-only', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--input-dim', type=int, default=7, choices=(5,7), help='Before optional momentum features')
    parser.add_argument('--output-dim', type=int, default=5, help='Before tracker/calo energy outputs')
    parser.add_argument('--momentum', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--momentum-amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--calo-head', action=argparse.BooleanOptionalAction, default=True,
                        help='Without calo head use seed energy for neutral clusters')
    parser.add_argument('--model-variant', choices=('auto','legacy','multihead'), default='auto')
    parser.add_argument('--regression-output-activation', choices=('current','linear'), default='current',
                        help='Explicit linear output for old multihead checkpoints trained without Softplus; '
                             'current preserves the installed model definition')
    parser.add_argument('--tbeta', type=float, default=.9)
    parser.add_argument('--td', type=float, default=.5)
    parser.add_argument('--energy-policy', choices=('alpha','any-track'), default='alpha')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=-1, help='Exclusive input index; -1 means all')
    parser.add_argument('--exclude-gap-hits', action='store_true',
                        help='Remove only ECAL GapHits rows before featurization and model forward; retain full Pandora PFOs')
    parser.add_argument('--detail', choices=('full','clusters','events'), default='full',
                        help='clusters/events omit per-hit and efficiency/purity tables to reduce full-sample output size')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--keep-failed-events', action='store_true',
                        help='Record failed inference with status and NaN instead of aborting')
    parser.add_argument('--replace', action='store_true')
    write(parser.parse_args())

if __name__ == '__main__':
    main()
