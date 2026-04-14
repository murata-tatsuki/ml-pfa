import uproot
import numpy as np

f = uproot.open("output.root")
t = f["t"].arrays(library="np")   # arbre MC particles
r = f["reco"].arrays(library="np") # arbre clusters reco

# Efficacité (par particule MC)
eff = t["edep_match"] / t["edep"]

# Pureté (par cluster reco)
pur = r["edep_match"] / r["edep_reco"]

# Filtrer par espèce (PDG codes)
# pion chargé = ±211, kaon chargé = ±321, photon = 22, neutron = 2112, K0 = 130/310
mask_pion = np.abs(t["mcpdg"]) == 211
mask_kaon = np.abs(t["mcpdg"]) == 321
mask_electron = np.abs(t["mcpdg"]) == 11
mask_photon = t["mcpdg"] == 22
mask_neutron = t["mcpdg"] == 2112
mask_k0 = np.isin(np.abs(t["mcpdg"]), [130, 310])
mask_k0_reco = np.isin(np.abs(r["mcpdg_match"]), [130, 310])  # pour pureté K0


print("Efficacité pion :", np.nanmean(eff[mask_pion]))
print("Efficacité kaon :", np.nanmean(eff[mask_kaon]))
print("Efficacité électron :", np.nanmean(eff[mask_electron]))
print("Efficacité photon :", np.nanmean(eff[mask_photon]))
print("Efficacité neutron :", np.nanmean(eff[mask_neutron]))
print("Efficacité K0 :", np.nanmean(eff[mask_k0]))