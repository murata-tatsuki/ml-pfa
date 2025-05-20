import json
import h5py
import awkward as ak
import numpy as np

def load_awkward(filename):
    file = h5py.File(filename)
    group = file["awkward"]
    
    form = ak.forms.from_json(group.attrs["form"])
    length = json.loads(group.attrs["length"])
    container = {k: np.asarray(v) for k, v in group.items()}

    return ak.from_buffers(form, length, container)    


def load_awkward2(filename):
    file = h5py.File(filename)

    feat = file["feature"]
    form = ak.forms.from_json(feat.attrs["form"])
    length = json.loads(feat.attrs["length"])
    container = {k: np.asarray(v) for k, v in feat.items()}
    ak_feat = ak.from_buffers(form, length, container)    

    label = file["label"]
    form = ak.forms.from_json(label.attrs["form"])
    length = json.loads(label.attrs["length"])
    container = {k: np.asarray(v) for k, v in label.items()}
    ak_label = ak.from_buffers(form, length, container)    

    ak_pred = None
    ak_energy = None
    ak_pandora = None
    ak_cluster = None
    ak_event = None


    if "cluster" in file:        
        clus = file["cluster"]
        form = ak.forms.from_json(clus.attrs["form"])
        length = json.loads(clus.attrs["length"])
        container = {k: np.asarray(v) for k, v in clus.items()}
        ak_cluster = ak.from_buffers(form, length, container)
    if "event" in file:        
        event = file["event"]
        form = ak.forms.from_json(event.attrs["form"])
        length = json.loads(event.attrs["length"])
        container = {k: np.asarray(v) for k, v in event.items()}
        ak_event = ak.from_buffers(form, length, container)
    if "pandora" in file:
        pand = file["pandora"]
        form = ak.forms.from_json(pand.attrs["form"])
        length = json.loads(pand.attrs["length"])
        container = {k: np.asarray(v) for k, v in pand.items()}
        ak_pandora = ak.from_buffers(form, length, container)

    if "pred" in file:        
        pred = file["pred"]
        form = ak.forms.from_json(pred.attrs["form"])
        length = json.loads(pred.attrs["length"])
        container = {k: np.asarray(v) for k, v in pred.items()}
        ak_pred = ak.from_buffers(form, length, container)    

    if "energy" in file:
        energy = file["energy"]
        form = ak.forms.from_json(energy.attrs["form"])
        length = json.loads(energy.attrs["length"])
        container = {k: np.asarray(v) for k, v in energy.items()}
        ak_energy = ak.from_buffers(form, length, container) 

    return ak_feat, ak_label, ak_pred, ak_energy, ak_pandora, ak_cluster, ak_event

def load_awkwards(filenames):
    print(f"{filenames=}")
    assert(len(filenames)>0)
    feats_list, labels_list = [], []
    for i, file in enumerate(filenames):
        print(f"Reading file: {file=}")
        feat, label, _, _, _, _, _ = load_awkward2(file)
        # if i==0:
        #     ak_feats = feat
        #     ak_labels = label
        # else:
        #     ak_feats = ak.concatenate((ak_feats, feat), axis=0)
        #     ak_labels = ak.concatenate((ak_labels, label), axis=0)
        feats_list.append(feat)
        labels_list.append(label)

        #print (ak.num(ak_feats,axis=0), ak.num(ak_labels,axis=0))
    # start = time()
    ak_feats = ak.concatenate(feats_list, axis=0)
    ak_labels = ak.concatenate(labels_list, axis=0)
    # print(time()-start)

    return ak_feats, ak_labels

def load_awkwards_all(filenames):
    print(f"{filenames=}")
    assert(len(filenames)>0)
    feats_list, labels_list, preds_list, energys_list, pandoras_list, clusters_list, events_list = [], [], [], [], [], [], []
    for i, file in enumerate(filenames):
        print(f"Reading file: {file=}")
        feat, label, pred, energy, pandora, cluster, event = load_awkward2(file)
        # if i==0:
        #     ak_feats = feat
        #     ak_labels = label
        # else:
        #     ak_feats = ak.concatenate((ak_feats, feat), axis=0)
        #     ak_labels = ak.concatenate((ak_labels, label), axis=0)
        feats_list.append(feat)
        labels_list.append(label)
        if pred is not None: preds_list.append(pred)
        if energy is not None: energys_list.append(energy)
        if pandora is not None: pandoras_list.append(pandora)
        if cluster is not None: clusters_list.append(cluster)
        if event is not None: events_list.append(event)

        #print (ak.num(ak_feats,axis=0), ak.num(ak_labels,axis=0))
    # start = time()
    ak_feats = ak.concatenate(feats_list, axis=0)
    ak_labels = ak.concatenate(labels_list, axis=0)
    ak_preds = ak.concatenate(preds_list, axis=0) if preds_list else None
    ak_energys = ak.concatenate(energys_list, axis=0) if energys_list else None
    ak_pandoras = ak.concatenate(pandoras_list, axis=0) if pandoras_list else None
    ak_clusters = ak.concatenate(clusters_list, axis=0) if clusters_list else None
    ak_events = ak.concatenate(events_list, axis=0) if events_list else None
    # print(time()-start)

    return ak_feats, ak_labels, ak_preds, ak_energys, ak_pandoras, ak_clusters, ak_events

def load_awkward2_pandora(filename):
    file = h5py.File(filename)
    pand = file["pandora"]
    
    form = ak.forms.from_json(pand.attrs["form"])
    length = json.loads(pand.attrs["length"])
    container = {k: np.asarray(v) for k, v in pand.items()}

    ak_pandora = ak.from_buffers(form, length, container)    

    return ak_pandora

def load_awkwards_pandora(filenames):
    print(f"{filenames=}")
    assert(len(filenames)>0)
    pands_list = []
    for i, file in enumerate(filenames):
        print(f"Reading file: {file=}")
        pand = load_awkward2_pandora(file)
        pands_list.append(pand)
        # if i==0:
        #     ak_pand = pand
        # else:
        #     ak_pand = ak.concatenate((ak_pand, pand), axis=0)
        #print (ak.num(ak_feats,axis=0), ak.num(ak_labels,axis=0))
    ak_pand = ak.concatenate(pands_list, axis=0)
    return ak_pand

def save_awkward(filename, ak_feat, ak_label, ak_pred = None, ak_energy = None, ak_x = None, ak_y = None, ak_pandora = None, ak_eventEnergy = None, ak_bremsConversion = None):
    file = h5py.File(filename,"w")
    g_feat = file.create_group("feature")

    form, length, container = ak.to_buffers(ak_feat, container=g_feat)
    g_feat.attrs["form"] = form.to_json()
    g_feat.attrs["length"] = json.dumps(length)

    g_label = file.create_group("label")

    form, length, container = ak.to_buffers(ak_label, container=g_label)
    g_label.attrs["form"] = form.to_json()
    g_label.attrs["length"] = json.dumps(length)

    if ak_pred is not None:
        g_pred = file.create_group("pred")

        form, length, container = ak.to_buffers(ak_pred, container=g_pred)
        g_pred.attrs["form"] = form.to_json()
        g_pred.attrs["length"] = json.dumps(length)

    if ak_energy is not None:
        g_energy = file.create_group("energy")

        form, length, container = ak.to_buffers(ak_energy, container=g_energy)
        g_energy.attrs["form"] = form.to_json()
        g_energy.attrs["length"] = json.dumps(length)

    if ak_x is not None:
        g_x = file.create_group("x")

        form, length, container = ak.to_buffers(ak_x, container=g_x)
        g_x.attrs["form"] = form.to_json()
        g_x.attrs["length"] = json.dumps(length)

    if ak_y is not None:
        g_y = file.create_group("y")

        form, length, container = ak.to_buffers(ak_y, container=g_y)
        g_y.attrs["form"] = form.to_json()
        g_y.attrs["length"] = json.dumps(length)

    if ak_pandora is not None:
        g_pand = file.create_group("pandora")

        form, length, container = ak.to_buffers(ak_pandora, container=g_pand)
        g_pand.attrs["form"] = form.to_json()
        g_pand.attrs["length"] = json.dumps(length)

    if ak_eventEnergy is not None:
        g_event = file.create_group("event")

        form, length, container = ak.to_buffers(ak_eventEnergy, container=g_event)
        g_event.attrs["form"] = form.to_json()
        g_event.attrs["length"] = json.dumps(length)

    if ak_bremsConversion is not None:
        g_cluster = file.create_group("cluster")

        form, length, container = ak.to_buffers(ak_bremsConversion, container=g_cluster)
        g_cluster.attrs["form"] = form.to_json()
        g_cluster.attrs["length"] = json.dumps(length)
