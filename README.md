# `jetclass-qcd-top`

This repo contains the code to filter the 
[JetClass dataset](https://github.com/jet-universe/particle_transformer) 
for QCD/top jets and add the ParT predictions.
    
The code is not refined yet, but gets the job done for now.

The resulting datasets are stored on [DESY Sync&Share](https://syncandshare.desy.de/index.php/s/Fx9W8Q4bgmN7HpQ).
Each dataset contains approximately the same number of (hadronic) top-jets
and QCD jets.

| Filename | Number of jets |
| --- | --- |
| `filtered_jetclass_train.h5` | 4M |
| `filtered_jetclass_val.h5` | 1M |
| `filtered_jetclass_test.h5` | 1M |

## Downloading and reading the files

You can either download the files via the web interface or via the command line.
In the command line you can use
```bash
wget https://syncandshare.desy.de/index.php/s/AXYmk4NwrLdJQgR/download/filtered_jetclass_train.h5
wget https://syncandshare.desy.de/index.php/s/AXYmk4NwrLdJQgR/download/filtered_jetclass_val.h5
wget https://syncandshare.desy.de/index.php/s/AXYmk4NwrLdJQgR/download/filtered_jetclass_test.h5
```
After [downloading the files](https://syncandshare.desy.de/index.php/s/Fx9W8Q4bgmN7HpQ), 
you can simply load the files using e.g. `pandas` or `h5py`.
The code snippet below assumes you are in the directory where the files are located.

```py
import pandas as pd

df_train = pd.read_hdf("filtered_jetclass_train.h5", key="df")
df_val = pd.read_hdf("filtered_jetclass_val.h5", key="df")
df_test = pd.read_hdf("filtered_jetclass_test.h5", key="df")
```
    
## Content of the output files
The files contain the jet-level features as well as the (rescaled ParT top quark)
prediction.

| Variable name | Description |
| --- | --- |
| `jet_p_top_kin = p_Tbqq / (p_Tbqq + p_QCD)` | Rescaled top quark probability of [ParT-kin](https://github.com/jet-universe/particle_transformer/blob/main/data/JetClass/JetClass_kin.yaml) |
| `label_top` | `label_top=1` for top jets and `label_top=0` for QCD jets |
| `jet_pt` | |
| `jet_eta` | |
| `jet_phi` | |
| `jet_energy` | |
| `jet_nparticles` | |
| `jet_sdmass` | |
| `jet_tau1` | |
| `jet_tau2` | |
| `jet_tau3` | |
| `jet_tau4` | |
| `aux_genpart_eta` | |
| `aux_genpart_phi` | |
| `aux_genpart_pid` | |
| `aux_genpart_pt` | |
| `aux_truth_match` | |

**To see the full comparison of variables between the original JetClass dataset
and the filtered dataset, click on the arrow below.**

<details>
  <summary>Overview / variable comparison to original JetClass dataset</summary>

| Variable name | Included / removed / added |
| --- | --- |
| `label_top` | Added |
| `jet_p_top_kin` | Added |
| `part_px` | Removed |
| `part_py` | Removed |
| `part_pz` | Removed |
| `part_energy` | Removed |
| `part_deta` | Removed |
| `part_dphi` | Removed |
| `part_d0val` | Removed |
| `part_d0err` | Removed |
| `part_dzval` | Removed |
| `part_dzerr` | Removed |
| `part_charge` | Removed |
| `part_isChargedHadron` | Removed |
| `part_isNeutralHadron` | Removed |
| `part_isPhoton` | Removed |
| `part_isElectron` | Removed |
| `part_isMuon` | Removed |
| `label_QCD` | Removed |
| `label_Hbb` | Removed |
| `label_Hcc` | Removed |
| `label_Hgg` | Removed |
| `label_H4q` | Removed |
| `label_Hqql` | Removed |
| `label_Zqq` | Removed |
| `label_Wqq` | Removed |
| `label_Tbqq` | Removed |
| `label_Tbl` | Removed |
| `jet_pt` | Included |
| `jet_eta` | Included |
| `jet_phi` | Included |
| `jet_energy` | Included |
| `jet_nparticles` | Included |
| `jet_sdmass` | Included |
| `jet_tau1` | Included |
| `jet_tau2` | Included |
| `jet_tau3` | Included |
| `jet_tau4` | Included |
| `aux_genpart_eta` | Included |
| `aux_genpart_phi` | Included |
| `aux_genpart_pid` | Included |
| `aux_genpart_pt` | Included |
| `aux_truth_match` | Included |
</details>
    
## Run the code (on the DESY Maxwell cluster)
You'll have to make sure that you have the JetClass dataset stored on your
machine and adapt the paths in `prepare_dataset.py` accordingly.

The code can then be executed within this repo by running the following
singularity command:
```bash
singularity exec --nv -B /home -B /beegfs /beegfs/desy/user/birkjosc/singularity_images/pytorch-image-v0.0.8.img \
    bash -c "source /opt/conda/bin/activate && python prepare_dataset.py"
```
If you don't have access to the DESY Maxwell cluster, you can also run the code
somewhere else of course, but you'll have to build the singularity image
yourself. The image is located on DockerHub at `jobirk/pytorch-image:v0.0.8`.
