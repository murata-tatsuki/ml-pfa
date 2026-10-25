"""Single-epoch ROOT identities and lossless, event-safe legacy consolidation.

Only the five latest Cedric trees are copied. Compressed baskets are retained;
the integer event branch is rebuilt in C++ to make event IDs unique across H5s.
The public save_root implementation and its frozen source identity stay intact.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

METADATA_KEY = 'epoch_scan_metadata'
SCHEMA = 'epoch-root-1'
TREE_NAMES = ('t', 'reco', 'prediction', 'event', 'jet')
_CPP_LOADED = False


def input_identity(item, check=False):
    """Normalize the planner and writer spellings of an original H5 identity."""
    path = Path(item['path']).expanduser().resolve()
    result = dict(path=str(path), size=int(item['size']),
                  mtime_ns=int(item.get('mtime_ns', item.get('modified_ns'))))
    if check:
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (result['size'], result['mtime_ns']):
            raise ValueError('Input H5 changed: ' + str(path))
    return result


def _inputs(items, check=False):
    result = [input_identity(item, check) for item in items]
    if not result or len({item['path'] for item in result}) != len(result):
        raise ValueError('Epoch output requires distinct, ordered H5 inputs')
    return result


@contextmanager
def _root_file(path, mode='READ'):
    """Open a ROOT file without leaving its directory selected after closing."""
    import ROOT

    if mode in ('READ', 'UPDATE') and not Path(path).is_file():
        raise OSError('Missing ROOT file: ' + str(path))
    with ROOT.TDirectory.TContext():
        handle = ROOT.TFile.Open(str(path), mode)
        try:
            if not handle or handle.IsZombie():
                raise OSError('Cannot open ROOT file (' + mode + '): ' + str(path))
            yield handle
        finally:
            if handle:
                handle.Close()


def _json_object(source, name, required=True):
    item = source.Get(name)
    if not item:
        if required:
            raise ValueError('Missing ROOT metadata: ' + name)
        return None
    if not item.InheritsFrom('TObjString'):
        raise ValueError('Invalid ROOT metadata type: ' + name)
    value = json.loads(str(item.GetString()))
    if not isinstance(value, dict):
        raise ValueError('Invalid ROOT metadata: ' + name)
    return value


def _tree_entries(source):
    counts = {}
    for name in TREE_NAMES:
        tree = source.Get(name)
        if not tree or not tree.InheritsFrom('TTree'):
            raise ValueError('Missing ROOT TTree: ' + name)
        counts[name] = int(tree.GetEntries())
    return counts


def _validate_root(*args, **kwargs):
    import ROOT
    from root_export import cedric

    with ROOT.TDirectory.TContext():
        return cedric._validate_root(*args, **kwargs)


def _ranges(metadata, inputs=None):
    if metadata.get('schema') != SCHEMA or metadata.get('event_range_policy') != 'per-input':
        raise ValueError('Unsupported single-epoch ROOT identity')
    ranges = metadata.get('source_ranges')
    if not isinstance(ranges, list) or not ranges:
        raise ValueError('Missing epoch source ranges')
    offset, normalized = 0, []
    for item in ranges:
        count = item['event_count']
        if (type(count) is not int or count < 0
                or type(item['event_offset']) is not int or item['event_offset'] != offset):
            raise ValueError('Invalid or noncontiguous epoch source ranges')
        normalized.append(input_identity(item['input']))
        offset += count
    if offset > 2147483647:
        raise ValueError('Epoch event IDs exceed the Cedric Int_t schema')
    if inputs is not None and normalized != _inputs(inputs):
        raise ValueError('Epoch source ranges differ from planned inputs')
    if len({item['path'] for item in normalized}) != len(normalized):
        raise ValueError('Duplicate epoch source H5')
    if metadata.get('tree_entries', {}).get('event') != offset:
        raise ValueError('Epoch event count differs from source ranges')
    return ranges


def read_epoch_metadata(path):
    """Return validated completion metadata, raising on absent/incomplete files."""
    with _root_file(path) as source:
        metadata = _json_object(source, METADATA_KEY)
        _ranges(metadata)
        actual = _tree_entries(source)
        if metadata.get('tree_entries') != actual:
            raise ValueError('Epoch tree entries differ from completion metadata')
        saved = _json_object(source, 'save_root_metadata')
        if saved.get('tree_entries') != actual:
            raise ValueError('Writer tree entries differ from epoch completion metadata')
        inputs = [item['input'] for item in metadata['source_ranges']]
        if [input_identity(item) for item in saved['signature']['inputs']] != _inputs(inputs):
            raise ValueError('Writer input identity differs from epoch source ranges')
        if metadata.get('signature') != saved['signature']:
            raise ValueError('Epoch and writer computation identities differ')
        particle = _json_object(source, 'particle_eval_metadata', required=(
            saved['signature']['settings'].get('pid_readout') == 'representative'))
        if particle is not None:
            _check_particle_identity(particle, saved, _inputs(inputs))
    return metadata


def validate_epoch_output(path, args, inputs):
    """Validate a matching epoch result before model construction or publication."""
    try:
        expected = _inputs(inputs, check=True)
        metadata = read_epoch_metadata(path)
        _ranges(metadata, expected)
        from root_export import cedric, runtime
        signature = runtime.output_signature(args, args.datapath)
        if metadata['signature'] != signature:
            return False
        import ROOT
        with ROOT.TDirectory.TContext():
            return cedric._validate_pid_resume(
                Path(path), args, signature['checkpoint_sha256']).valid
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
        return False


def write_epoch_metadata(path, source_ranges, mode='direct', sources=None):
    """Append the source map after the ordinary writer completion identity."""
    import ROOT

    with _root_file(path, 'UPDATE') as handle:
        saved = _json_object(handle, 'save_root_metadata')
        if saved.get('tree_entries') != _tree_entries(handle):
            raise ValueError('Writer tree entries differ from ROOT contents')
        metadata = dict(schema=SCHEMA, event_range_policy='per-input',
                        source_ranges=copy.deepcopy(source_ranges),
                        tree_entries=saved['tree_entries'], signature=saved['signature'],
                        provenance=dict(mode=mode))
        if sources is not None:
            metadata['provenance']['sources'] = sources
        _ranges(metadata, saved['signature']['inputs'])
        handle.cd()
        if ROOT.TObjString(json.dumps(metadata, sort_keys=True, allow_nan=False)).Write(
                METADATA_KEY, ROOT.TObject.kOverwrite) <= 0:
            raise OSError('Failed to write epoch completion identity')
    return metadata


def _file_identity(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def _ensure_cpp():
    global _CPP_LOADED
    if _CPP_LOADED:
        return
    import ROOT
    if not ROOT.gInterpreter.Declare(r'''
#ifndef MURATA_EPOCH_ROOT_EVENT_REBASE_V2
#define MURATA_EPOCH_ROOT_EVENT_REBASE_V2
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
namespace epoch_root_v2 {
void rebase(TTree* output, const std::vector<std::string>& paths,
            const std::string& name, const std::vector<long long>& offsets,
            const std::vector<long long>& counts) {
    if (!output || paths.size()!=offsets.size() || paths.size()!=counts.size())
        throw std::runtime_error("Invalid event rebase request");
    auto* destination = output->GetBranch("event");
    if (!destination) throw std::runtime_error("Missing output event branch");
    const auto entries = output->GetEntries();
    destination->Reset();
    Int_t global = 0;
    destination->SetAddress(&global);
    Long64_t written = 0;
    for (std::size_t index=0; index<paths.size(); ++index) {
        std::unique_ptr<TFile> file(TFile::Open(paths[index].c_str(), "READ"));
        if (!file || file->IsZombie()) throw std::runtime_error("Cannot reopen source ROOT");
        auto* tree = dynamic_cast<TTree*>(file->Get(name.c_str()));
        auto* branch = tree ? tree->GetBranch("event") : nullptr;
        if (!branch) throw std::runtime_error("Missing source event branch");
        Int_t local = 0;
        branch->SetAddress(&local);
        Int_t previous = -1;
        for (Long64_t row=0; row<tree->GetEntries(); ++row) {
            if (branch->GetEntry(row)<=0) throw std::runtime_error("Cannot read source event ID");
            if (local<0 || local>=counts[index] || local<previous ||
                (name=="event" && local!=row))
                throw std::runtime_error("Invalid source event order or range");
            previous=local;
            global=static_cast<Int_t>(offsets[index]+local);
            if (destination->Fill()<0) throw std::runtime_error("Cannot write rebased event ID");
            ++written;
        }
        branch->ResetAddress();
    }
    if (written!=entries || destination->GetEntries()!=entries)
        throw std::runtime_error("Rebased event entry count differs");
    output->SetEntries(entries);
    destination->ResetAddress();
}
void validate_events(TTree* tree, Long64_t count) {
    if (!tree || tree->GetEntries()!=count)
        throw std::runtime_error("Merged event tree count differs");
    auto* branch = tree->GetBranch("event");
    auto* leaf = branch ? branch->GetLeaf("event") : nullptr;
    if (!leaf || std::string(leaf->GetTypeName())!="Int_t" ||
        leaf->GetLeafCount() || leaf->GetLenStatic()!=1 || branch->GetEntries()!=count)
        throw std::runtime_error("Merged event IDs require a scalar Int_t branch");
    Int_t event = -1;
    branch->SetAddress(&event);
    try {
        for (Long64_t row=0; row<count; ++row) {
            if (branch->GetEntry(row)<=0 || event!=row)
                throw std::runtime_error("Merged global event IDs are not contiguous");
        }
    } catch (...) {
        branch->ResetAddress();
        throw;
    }
    branch->ResetAddress();
}
}
#endif
'''):
        raise RuntimeError('Cannot compile epoch event rebasing helper')
    ROOT.epoch_root_v2.rebase.__release_gil__ = True
    ROOT.epoch_root_v2.validate_events.__release_gil__ = True
    _CPP_LOADED = True


def _source_metadata(path, expected_input):
    with _root_file(path) as source:
        saved = _json_object(source, 'save_root_metadata')
        signature = saved.get('signature', {})
        if signature.get('version') != 'cedric-unified-1':
            raise ValueError('Unsupported source writer identity: ' + str(path))
        if [input_identity(item) for item in signature.get('inputs', [])] != [expected_input]:
            raise ValueError('Source ROOT input differs from expected H5: ' + str(path))
        counts = _tree_entries(source)
        if saved.get('tree_entries') != counts:
            raise ValueError('Source ROOT has incomplete tree counts: ' + str(path))
        pid = signature.get('settings', {}).get('pid_readout') == 'representative'
        particle = _json_object(source, 'particle_eval_metadata', required=pid)
    check = _validate_root(path, pid=pid)
    if not check.valid:
        raise ValueError('Incomplete source ROOT: ' + str(path) + ': ' + check.reason)
    if particle is not None:
        _check_particle_identity(particle, saved, [expected_input])
    return saved, particle


def _check_particle_identity(particle, saved, inputs):
    signature, counts = saved['signature'], saved['tree_entries']
    resume = particle.get('resume', {})
    if (particle.get('schema') != 'cedric-pid-root-1'
            or particle.get('checkpoint_sha256') != signature.get('checkpoint_sha256')
            or resume.get('version') != 1 or resume.get('tree_entries') != counts
            or [input_identity(item) for item in resume.get('inputs', [])] != inputs
            or set(particle.get('inputs', {})) != {item['path'] for item in inputs}):
        raise ValueError('PID completion/provenance differs from writer identity')
    if particle.get('code_sha256') != signature.get('implementation', {}).get('root_export/cedric.py'):
        raise ValueError('PID writer source differs from computation identity')
    from root_export.cedric import _PID_RESUME_FIELDS
    arguments, settings = particle.get('arguments', {}), signature['settings']
    for name in (*_PID_RESUME_FIELDS, 'tbeta', 'td'):
        if arguments.get(name) != settings.get(name):
            raise ValueError('PID arguments differ from writer identity: ' + name)


def _merge_metadata(saved, particles, inputs, output):
    signature = copy.deepcopy(saved[0]['signature'])
    reference = {key: value for key, value in signature.items() if key != 'inputs'}
    for item in saved[1:]:
        if {key: value for key, value in item['signature'].items() if key != 'inputs'} != reference:
            raise ValueError('Source checkpoint, physics or implementation identities differ')
    signature['inputs'] = [dict(path=item['path'], size=item['size'], modified_ns=item['mtime_ns'])
                           for item in inputs]
    counts = {name: sum(item['tree_entries'][name] for item in saved) for name in TREE_NAMES}
    aggregate = dict(signature=signature, tree_entries=counts)
    particle = None
    if any(item is not None for item in particles):
        if any(item is None for item in particles):
            raise ValueError('Sources disagree on PID metadata availability')
        policy = lambda item: {key: value for key, value in item.items()
                               if key not in ('arguments', 'inputs', 'resume')}
        if any(policy(item) != policy(particles[0]) for item in particles[1:]):
            raise ValueError('Source PID/energy readout policies or trained configurations differ')
        particle = copy.deepcopy(particles[0])
        particle['inputs'] = {key: value for item in particles for key, value in item['inputs'].items()}
        particle['arguments'].update(datapath=','.join(item['path'] for item in inputs),
                                     outfile=str(output), multi_h5=False, replace=False)
        particle['resume'] = dict(version=1, inputs=copy.deepcopy(inputs), tree_entries=counts)
    return aggregate, particle


def merge_roots(sources, output, inputs, replace=False):
    """Consolidate per-H5 outputs without inference; publish only a valid result.

    Sources and inputs must have the same deterministic order. Callers retain
    ownership of source files and serialize writes to this output with scan
    locks. Source ROOTs are never modified or deleted by this function.
    """
    import ROOT

    ROOT.gROOT.SetBatch(True)
    sources = [Path(path).expanduser().resolve() for path in sources]
    output = Path(output).expanduser()
    if output.is_symlink() or output.is_dir():
        raise ValueError('Output must be a regular file path: ' + str(output))
    output = output.absolute()
    inputs = _inputs(inputs, check=True)
    if len(sources) != len(inputs) or len(set(sources)) != len(sources):
        raise ValueError('Require one distinct ROOT per ordered input H5')
    if output.resolve() in set(sources) | {Path(item['path']) for item in inputs}:
        raise ValueError('Output must differ from every source ROOT and H5')
    identities = [_file_identity(path) for path in sources]
    saved, particles = [], []
    for path, identity in zip(sources, inputs):
        ordinary, particle = _source_metadata(path, identity)
        saved.append(ordinary)
        particles.append(particle)
    aggregate, particle = _merge_metadata(saved, particles, inputs, output)
    ranges, offset = [], 0
    for item, identity in zip(saved, inputs):
        count = item['tree_entries']['event']
        ranges.append(dict(input=identity, event_offset=offset, event_count=count))
        offset += count
    provisional = dict(schema=SCHEMA, event_range_policy='per-input', source_ranges=ranges,
                       tree_entries=aggregate['tree_entries'])
    _ranges(provisional, inputs)
    pid = aggregate['signature']['settings']['pid_readout'] == 'representative'
    if output.exists() and not replace:
        try:
            existing = read_epoch_metadata(output)
            if (existing['signature'] == aggregate['signature']
                    and existing['source_ranges'] == ranges
                    and _validate_root(output, pid=pid).valid):
                return existing
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.' + output.name + '.part.' + uuid.uuid4().hex + '.root')
    try:
        _ensure_cpp()
        with _root_file(staging, 'RECREATE') as handle:
            for name in TREE_NAMES:
                chain = ROOT.TChain(name)
                for path in sources:
                    if chain.Add(str(path)) != 1:
                        raise OSError('Cannot add source tree: ' + str(path))
                handle.cd()
                clone = chain.CloneTree(-1, 'fast')
                if not clone or int(clone.GetEntries()) != aggregate['tree_entries'][name]:
                    raise RuntimeError('Fast tree copy failed: ' + name)
                ROOT.epoch_root_v2.rebase(clone, [str(path) for path in sources], name,
                    [item['event_offset'] for item in ranges],
                    [item['event_count'] for item in ranges])
                handle.cd()
                if clone.Write('', ROOT.TObject.kOverwrite) <= 0:
                    raise OSError('Cannot publish copied tree: ' + name)
                # The chain is destroyed before advancing to another tree; all
                # data addresses have been detached from the copied event branch.
                del clone, chain
            for name, value in (('save_root_metadata', aggregate),
                                ('particle_eval_metadata', particle)):
                if value is not None:
                    handle.cd()
                    if ROOT.TObjString(json.dumps(value, sort_keys=True, allow_nan=False)).Write(
                            name, ROOT.TObject.kOverwrite) <= 0:
                        raise OSError('Cannot write merged completion metadata: ' + name)
        metadata = write_epoch_metadata(staging, ranges, mode='merged', sources=identities)
        checked = read_epoch_metadata(staging)
        if checked != metadata:
            raise ValueError('Epoch completion metadata did not round-trip')
        expected = aggregate['tree_entries']
        check = _validate_root(staging,
            tuple(expected[name] for name in ('event', 'prediction', 'jet')), pid=pid)
        if not check.valid:
            raise ValueError('Merged ROOT failed structural validation: ' + check.reason)
        with _root_file(staging) as source:
            ROOT.epoch_root_v2.validate_events(source.Get('event'), offset)
        if [_file_identity(path) for path in sources] != identities:
            raise ValueError('Source ROOT changed while consolidating')
        _inputs(inputs, check=True)
        os.replace(staging, output)
        return metadata
    finally:
        staging.unlink(missing_ok=True)
