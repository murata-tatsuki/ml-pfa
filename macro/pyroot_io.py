"""Small, shared PyROOT readers and bulk numeric TTree writer.

Returned NumPy values own their storage, including collection-valued branches,
so callers may close the source TFile immediately after reading. ROOT performs
all tree iteration in C++; Python copies the returned columns.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
from pathlib import Path

import numpy as np


_SCALAR_DTYPES = {
    "Bool_t": np.bool_, "bool": np.bool_,
    "Char_t": np.int8, "char": np.int8, "signedchar": np.int8,
    "UChar_t": np.uint8, "unsignedchar": np.uint8,
    "Short_t": np.int16, "short": np.int16, "shortint": np.int16,
    "UShort_t": np.uint16, "unsignedshort": np.uint16,
    "unsignedshortint": np.uint16,
    "Int_t": np.int32, "int": np.int32, "signedint": np.int32,
    "UInt_t": np.uint32, "unsigned": np.uint32, "unsignedint": np.uint32,
    "Long_t": np.dtype("l"), "long": np.dtype("l"), "longint": np.dtype("l"),
    "ULong_t": np.dtype("L"), "unsignedlong": np.dtype("L"),
    "unsignedlongint": np.dtype("L"),
    "Long64_t": np.int64, "longlong": np.int64, "longlongint": np.int64,
    "ULong64_t": np.uint64, "unsignedlonglong": np.uint64,
    "unsignedlonglongint": np.uint64,
    "Float_t": np.float32, "Float16_t": np.float32, "float": np.float32,
    "Double_t": np.float64, "Double32_t": np.float64, "double": np.float64,
}


@contextmanager
def open_root(path, mode="READ"):
    """Open a local ROOT file and restore the previously selected directory."""
    import ROOT

    mode = str(mode).upper()
    if mode in ("READ", "UPDATE") and not Path(path).is_file():
        raise OSError("Missing ROOT file: " + str(path))
    with ROOT.TDirectory.TContext():
        handle = ROOT.TFile.Open(str(path), mode)
        try:
            if not handle or handle.IsZombie():
                raise OSError("Cannot open ROOT file (" + mode + "): " + str(path))
            yield handle
        finally:
            if handle:
                handle.Close()


def _object(directory, name):
    item = directory.Get(str(name))
    if not item:
        raise ValueError("Missing ROOT object: " + str(name))
    return item


def get_tree(directory, name):
    """Return a TTree, refusing missing objects and objects of another class."""
    item = _object(directory, name)
    if not item.InheritsFrom("TTree"):
        raise ValueError("ROOT object is not a TTree: " + str(name))
    return item


def get_directory(directory, name):
    item = _object(directory, name)
    if not item.InheritsFrom("TDirectory"):
        raise ValueError("ROOT object is not a TDirectory: " + str(name))
    return item


def has_branch(tree, name):
    return bool(tree.GetBranch(str(name)))


@contextmanager
def _readable_branches(tree, names):
    # TTreeReader otherwise silently sees stale values for disabled branches.
    required = set(names)
    for name in names:
        for leaf in tree.GetBranch(name).GetListOfLeaves():
            count = leaf.GetLeafCount()
            if count:
                required.add(str(count.GetBranch().GetName()))
    disabled = sorted(name for name in required if not tree.GetBranchStatus(name))
    try:
        for name in disabled:
            tree.SetBranchStatus(name, 1)
        yield
    finally:
        for name in disabled:
            tree.SetBranchStatus(name, 0)


def _column_type(tree, name):
    """Describe scalar, leaf-array and STL numeric branches without RDF JIT."""
    branch = tree.GetBranch(name)
    class_name = str(branch.GetClassName()).replace("std::", "").replace(" ", "")
    if class_name in ("string", "basic_string<char>"):
        return "string", None, False
    collection = False
    if class_name.startswith(("vector<", "ROOT::VecOps::RVec<")):
        kind = class_name.split("<", 1)[1].split(",", 1)[0].rstrip(">")
        collection = True
    elif class_name:
        raise ValueError("Unsupported ROOT branch class: " + name + " (" + class_name + ")")
    else:
        leaves = branch.GetListOfLeaves()
        if len(leaves) != 1:
            raise ValueError("ROOT array reader requires one leaf per branch: " + name)
        leaf = leaves.At(0)
        kind = str(leaf.GetTypeName()).replace("std::", "").replace(" ", "")
        collection = bool(leaf.GetLeafCount()) or "[" in str(leaf.GetTitle())
    if kind not in _SCALAR_DTYPES:
        raise ValueError("Unsupported ROOT numeric type: " + name + " (" + kind + ")")
    dtype = np.dtype(_SCALAR_DTYPES[kind])
    if kind in ("Long_t", "long", "longint"):
        root_type = "Long_t"
    elif kind in ("ULong_t", "unsignedlong", "unsignedlongint"):
        root_type = "ULong_t"
    else:
        root_type = {
            ("b", 1): "Bool_t", ("i", 1): "Char_t", ("u", 1): "UChar_t",
            ("i", 2): "Short_t", ("u", 2): "UShort_t", ("i", 4): "Int_t",
            ("u", 4): "UInt_t", ("i", 8): "Long64_t", ("u", 8): "ULong64_t",
            ("f", 4): "Float_t", ("f", 8): "Double_t",
        }[(dtype.kind, dtype.itemsize)]
    return root_type, dtype, collection


def _reader():
    import ROOT

    if not hasattr(ROOT, "pyroot_io_detail_v2"):
        if not ROOT.gInterpreter.Declare(r'''
            #include <TTree.h>
            #include <TTreeReader.h>
            #include <TTreeReaderArray.h>
            #include <TTreeReaderValue.h>
            #include <cstdint>
            #include <cstring>
            #include <memory>
            #include <stdexcept>
            #include <string>
            #include <vector>
            namespace pyroot_io_detail_v2 {
            struct Column {
                // uint64 storage exposes one stable contiguous NumPy view per
                // column. No per-entry RVec proxy or JIT conversion is needed.
                std::vector<ULong64_t> storage;
                std::vector<ULong64_t> offsets;
                std::vector<std::string> strings;
                ULong64_t byte_count = 0;
                ULong64_t data_address() const {
                    return static_cast<ULong64_t>(reinterpret_cast<std::uintptr_t>(storage.data()));
                }
                ULong64_t offsets_address() const {
                    return static_cast<ULong64_t>(reinterpret_cast<std::uintptr_t>(offsets.data()));
                }
                void* append_bytes(std::size_t count) {
                    const auto start = byte_count;
                    byte_count += count;
                    storage.resize((byte_count + sizeof(ULong64_t) - 1) / sizeof(ULong64_t));
                    if (storage.empty()) return nullptr;
                    return reinterpret_cast<unsigned char*>(storage.data()) + start;
                }
            };
            struct Reader {
                virtual ~Reader() = default;
                virtual void append(Column&) = 0;
            };
            template<class T> struct ScalarReader : Reader {
                TTreeReaderValue<T> value;
                ScalarReader(TTreeReader& reader, const std::string& name)
                    : value(reader, name.c_str()) {}
                void append(Column& column) override {
                    if (value.GetSetupStatus() < 0) throw std::runtime_error("Cannot bind scalar ROOT branch");
                    const T item = *value;
                    std::memcpy(column.append_bytes(sizeof(T)), &item, sizeof(T));
                }
            };
            template<class T> struct ArrayReader : Reader {
                TTreeReaderArray<T> value;
                ArrayReader(TTreeReader& reader, const std::string& name)
                    : value(reader, name.c_str()) {}
                void append(Column& column) override {
                    if (value.GetSetupStatus() < 0) throw std::runtime_error("Cannot bind array ROOT branch");
                    const auto count = value.GetSize();
                    auto* target = static_cast<unsigned char*>(column.append_bytes(count * sizeof(T)));
                    for (std::size_t i = 0; i < count; ++i) {
                        const T item = value[i];
                        std::memcpy(target + i * sizeof(T), &item, sizeof(T));
                    }
                    column.offsets.push_back(column.byte_count / sizeof(T));
                }
            };
            struct StringReader : Reader {
                TTreeReaderValue<std::string> value;
                StringReader(TTreeReader& reader, const std::string& name)
                    : value(reader, name.c_str()) {}
                void append(Column& column) override {
                    if (value.GetSetupStatus() < 0) throw std::runtime_error("Cannot bind string ROOT branch");
                    column.strings.push_back(*value);
                }
            };
            template<class T> std::unique_ptr<Reader> numeric_reader(
                    TTreeReader& reader, const std::string& name, bool collection) {
                if (collection) return std::unique_ptr<Reader>(new ArrayReader<T>(reader, name));
                return std::unique_ptr<Reader>(new ScalarReader<T>(reader, name));
            }
            std::unique_ptr<Reader> make_reader(TTreeReader& reader,
                    const std::string& name, const std::string& type, bool collection) {
                #define PYROOT_IO_READER(T) if (type == #T) return numeric_reader<T>(reader, name, collection);
                PYROOT_IO_READER(Bool_t)
                PYROOT_IO_READER(Char_t)
                PYROOT_IO_READER(UChar_t)
                PYROOT_IO_READER(Short_t)
                PYROOT_IO_READER(UShort_t)
                PYROOT_IO_READER(Int_t)
                PYROOT_IO_READER(UInt_t)
                PYROOT_IO_READER(Long_t)
                PYROOT_IO_READER(ULong_t)
                PYROOT_IO_READER(Long64_t)
                PYROOT_IO_READER(ULong64_t)
                PYROOT_IO_READER(Float_t)
                PYROOT_IO_READER(Double_t)
                #undef PYROOT_IO_READER
                if (type == "string") return std::unique_ptr<Reader>(new StringReader(reader, name));
                throw std::runtime_error("Unsupported ROOT reader type: " + type);
            }
            std::vector<Column> read(TTree* tree,
                    const std::vector<std::string>& names,
                    const std::vector<std::string>& types,
                    const std::vector<int>& collections) {
                TTreeReader reader(tree);
                std::vector<std::unique_ptr<Reader>> readers;
                std::vector<Column> columns(names.size());
                for (std::size_t i = 0; i < names.size(); ++i) {
                    readers.push_back(make_reader(reader, names[i], types[i], collections[i]));
                    if (collections[i]) columns[i].offsets.push_back(0);
                }
                Long64_t rows = 0;
                while (reader.Next()) {
                    for (std::size_t i = 0; i < names.size(); ++i) readers[i]->append(columns[i]);
                    ++rows;
                }
                if (rows != tree->GetEntries())
                    throw std::runtime_error("Incomplete ROOT TTree read: " + std::string(tree->GetName()));
                return columns;
            }
            }
        '''):
            raise RuntimeError("Cannot load PyROOT bulk TTree reader")
    return ROOT.pyroot_io_detail_v2.read


def arrays(tree, names=None):
    """Read selected columns together, in entry order, without repeated JIT.

    Scalars retain their native NumPy dtype; strings are Python str objects;
    numeric collections are object arrays of independent NumPy arrays. The
    C++ TTreeReader loop preserves entry order even with ROOT ImplicitMT on,
    and all returned values remain valid after closing the source file.
    """
    import ROOT

    if not tree or not tree.InheritsFrom("TTree"):
        raise ValueError("ROOT array input must be a TTree")
    names = ([str(branch.GetName()) for branch in tree.GetListOfBranches()]
             if names is None else list(map(str, names)))
    if len(set(names)) != len(names):
        raise ValueError("Duplicate ROOT branch names")
    missing = [name for name in names if not has_branch(tree, name)]
    if missing:
        raise ValueError(str(tree.GetName()) + ": missing branches " + str(missing))
    if not names:
        return {}
    descriptions = [_column_type(tree, name) for name in names]
    with ROOT.TDirectory.TContext(), _readable_branches(tree, names):
        try:
            columns = _reader()(tree, names, [item[0] for item in descriptions],
                                [int(item[2]) for item in descriptions])
        except Exception as error:
            raise ValueError("Cannot read ROOT TTree " + str(tree.GetName()) + ": " + str(error)) from error
        result = {}
        for index, (name, (kind, dtype, collection)) in enumerate(zip(names, descriptions)):
            column = columns[index]
            if kind == "string":
                owned = np.empty(len(column.strings), dtype=object)
                for position, value in enumerate(column.strings):
                    owned[position] = str(value)
            else:
                byte_count = int(column.byte_count)
                if byte_count:
                    # ROOT 6.28's vector/RVec __array_interface__ JITs an
                    # address-specific GetDataPointer expression per object.
                    # An integer address + ctypes view avoids accumulating
                    # executable mappings during scans with many epochs.
                    buffer = (ctypes.c_ubyte * byte_count).from_address(int(column.data_address()))
                    raw = np.ctypeslib.as_array(buffer)
                    values = raw.view(dtype).copy()
                else:
                    values = np.empty(0, dtype=dtype)
                if collection:
                    offset_buffer = (ctypes.c_uint64 * len(column.offsets)).from_address(
                        int(column.offsets_address()))
                    offsets = np.ctypeslib.as_array(offset_buffer).astype(np.intp, copy=True)
                    owned = np.empty(len(offsets) - 1, dtype=object)
                    for position in range(len(owned)):
                        owned[position] = values[offsets[position]:offsets[position + 1]].copy()
                else:
                    owned = values
            result[name] = owned
        return result


def read_json(directory, name, required=True):
    item = directory.Get(str(name))
    if not item:
        if not required:
            return {}
        raise ValueError("Missing ROOT JSON metadata: " + str(name))
    if not item.InheritsFrom("TObjString"):
        raise ValueError("ROOT JSON metadata is not a TObjString: " + str(name))
    try:
        value = json.loads(str(item.GetString()))
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid ROOT JSON metadata: " + str(name)) from error
    if not isinstance(value, dict):
        raise ValueError("ROOT metadata must contain a JSON object: " + str(name))
    return value


def read_title(directory, name):
    item = _object(directory, name)
    if not item.InheritsFrom("TNamed"):
        raise ValueError("ROOT object is not a TNamed: " + str(name))
    return str(item.GetTitle())


def read_value(directory, name):
    item = _object(directory, name)
    if not hasattr(item, "GetVal") or not str(item.ClassName()).startswith("TParameter<"):
        raise ValueError("ROOT object is not a TParameter: " + str(name))
    return item.GetVal()


def _writer():
    import ROOT

    if not hasattr(ROOT, "pyroot_io_detail_v1"):
        if not ROOT.gInterpreter.Declare(r'''
            #include <TBranch.h>
            #include <TDirectory.h>
            #include <TObject.h>
            #include <TTree.h>
            #include <array>
            #include <cstdint>
            #include <cstring>
            #include <stdexcept>
            #include <string>
            #include <vector>
            namespace pyroot_io_detail_v1 {
            void write(TDirectory* directory, const std::string& name,
                       const std::vector<std::string>& names,
                       const std::vector<ULong64_t>& addresses,
                       const std::vector<unsigned int>& sizes,
                       const std::vector<std::string>& leaf_types,
                       ULong64_t rows) {
                TDirectory::TContext context(directory);
                TTree tree(name.c_str(), name.c_str());
                tree.SetDirectory(directory);
                std::vector<std::array<unsigned char, 8>> buffers(names.size());
                for (std::size_t col = 0; col < names.size(); ++col) {
                    const std::string leaf = names[col] + "/" + leaf_types[col];
                    if (!tree.Branch(names[col].c_str(), buffers[col].data(), leaf.c_str()))
                        throw std::runtime_error("Cannot create ROOT branch: " + names[col]);
                }
                for (ULong64_t row = 0; row < rows; ++row) {
                    for (std::size_t col = 0; col < names.size(); ++col) {
                        const auto* source = reinterpret_cast<const unsigned char*>(
                            static_cast<std::uintptr_t>(addresses[col]));
                        std::memcpy(buffers[col].data(), source + row * sizes[col], sizes[col]);
                    }
                    if (tree.Fill() < 0) throw std::runtime_error("Cannot fill ROOT tree: " + name);
                }
                if (tree.Write(name.c_str(), TObject::kOverwrite) <= 0)
                    throw std::runtime_error("Cannot write ROOT tree: " + name);
            }
            }
        '''):
            raise RuntimeError("Cannot load PyROOT bulk TTree writer")
    return ROOT.pyroot_io_detail_v1.write


def write_tree(directory, name, columns):
    """Write scalar numeric NumPy columns, including zero-entry schemas.

    All rows are filled in C++, using the already-open caller's TDirectory.
    No temporary ROOT file or second handle to the output file is opened.
    """
    if not directory or not directory.InheritsFrom("TDirectory"):
        raise ValueError("ROOT output must be a TDirectory")
    output_file = directory.GetFile()
    if not output_file or not output_file.IsWritable():
        raise ValueError("ROOT output directory is not writable")
    name = str(name)
    if not name or "/" in name:
        raise ValueError("ROOT tree name must be a nonempty directory-local name")
    leaf_types = {("b", 1): "O", ("i", 1): "B", ("u", 1): "b",
                  ("i", 2): "S", ("u", 2): "s", ("i", 4): "I",
                  ("u", 4): "i", ("i", 8): "L", ("u", 8): "l",
                  ("f", 4): "F", ("f", 8): "D"}
    prepared, types = {}, []
    rows = None
    for branch, values in columns.items():
        branch = str(branch)
        if not branch or any(character in branch for character in "/[]:"):
            raise ValueError("Unsupported scalar ROOT branch name: " + branch)
        values = np.asarray(values)
        if values.ndim != 1 or (values.dtype.kind, values.dtype.itemsize) not in leaf_types:
            raise ValueError("ROOT writer requires one-dimensional scalar numeric columns: " + branch)
        if rows is not None and len(values) != rows:
            raise ValueError("ROOT writer columns have different lengths")
        rows = len(values)
        prepared[branch] = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("="))
        types.append(leaf_types[(values.dtype.kind, values.dtype.itemsize)])
    _writer()(directory, name, list(prepared),
              [int(values.ctypes.data) for values in prepared.values()],
              [int(values.dtype.itemsize) for values in prepared.values()], types, rows or 0)
