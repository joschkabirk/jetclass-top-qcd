"""Script to prepare a dataset with ParT model predictions and jet features from
the JetClass dataset."""
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from weaver.utils.dataset import SimpleIterDataset

from model.part import ParticleTransformerWrapper, part_default_kwargs

# NOTE: if you want to run this script, you have to adjust the parameters below
# TODO: make the parameters below command-line arguments
DEVELOPMENT_MODE = False  # if True, only a small subset of the data is used
DATA_CONFIG_JETCLASS = "data_config_full.yaml"
models_to_evaluate = {
    "ParT_full": {
        "model_path": (  # noqa: E501, fmt: off
            "/home/birkjosc/repositories/particle_transformer/models/ParT_full.pt"
        ),
        "input_dim": 17,
        "pf_features_indices": list(range(17)),  # use all features
    },
    "ParT_kin": {
        "model_path": (  # noqa: E501, fmt: off
            "/home/birkjosc/repositories/particle_transformer/models/ParT_kin.pt"
        ),
        "input_dim": 7,
        "pf_features_indices": [0, 1, 2, 3, 4, 15, 16],  # exclude the pid stuff
    },
}
JETCLASS_PATH = "/beegfs/desy/user/birkjosc/datasets/jetclass/JetClass"
PROCESSES = [
    # "HToBB",
    # "HToCC",
    # "HToGG",
    # "HToWW2Q1L",
    # "HToWW4Q",
    "TTBar",  # Top jets (hadronic decay)
    # "TTBarLep",
    # "WToQQ",
    "ZJetsToNuNu",  # QCD jets
    # "ZToQQ",
]
TRAIN_VAL_TEST_CONFIG = {
    "train": {
        "folder": "train_100M",
        "n_filtered": 4_000_000 if not DEVELOPMENT_MODE else 20_000,
    },
    "val": {
        "folder": "val_5M",
        "n_filtered": 1_000_000 if not DEVELOPMENT_MODE else 10_000,
    },
    "test": {
        "folder": "test_20M",
        "n_filtered": 4_000_000 if not DEVELOPMENT_MODE else 20_000,
    },
}
file_dict_jetclass = {
    split: {
        process: sorted(
            list(glob.glob(f"{JETCLASS_PATH}/{cfg['folder']}/{process}_*.root"))
        )[
            :20
        ]  # noqa: E501
        for process in PROCESSES
    }
    for split, cfg in TRAIN_VAL_TEST_CONFIG.items()
}
# these are the observer variables that are saved during evaluation
# they have to be defined in the data_config file under "observers"
VAR_NAMES_TO_SAVE = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_nparticles",
    "jet_sdmass",
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    "jet_tau4",
    "aux_genpart_eta",
    "aux_genpart_phi",
    "aux_genpart_pid",
    "aux_genpart_pt",
    "aux_truth_match",
]


def create_dataset_with_model_predictions(
    dataloader,
    models,
    input_names,
    output_file,
    n_stop=None,
    var_names_to_save=None,
    index_top=8,  # TODO: don't hardcode this --> extract from the data_config file
    index_qcd=0,  # TODO: don't hardcode this --> extract from the data_config file
):
    """Create a dataset with model predictions.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader that yields batches of data.
    models : dict
        Dict of models for which the predictions are calculated. The keys are
        the names of the models, the values are again dicts which contain the
        model path, the model itself (torch.nn.Module), input dimension and the
        indices of the particle features that are used as input.
    input_names : list of str
        Names of the inputs that are passed to the model.
    output_file : str
        Path to the output file.
    n_stop : int, optional
        Number of jets to evaluate (mostly useful for development).
        If None, all available jets are evaluated.
    var_names_to_save : list of str, optional
        Names of the observer variables that are saved in the output file.
    """
    var_names_to_save = [] if var_names_to_save is None else var_names_to_save
    data_dict = (
        {f"jet_p_top_{model_name}": [] for model_name in models.keys()}
        | {
            "label_top": [],
        }
        | {var_name: [] for var_name in var_names_to_save}
    )

    if n_stop is not None:
        n_batches = n_stop // dataloader.batch_size
        print(f"Evaluating {n_stop:,} jets. This will take {n_batches} batches.")

    with tqdm(dataloader) as tq:
        for i_batch, (X, y, observers) in enumerate(tq):
            if n_stop is not None:
                if i_batch * dataloader.batch_size >= n_stop:
                    print(f"Evaluated {n_stop} jets")
                    break
            inputs = [X[k].to("cuda") for k in input_names]
            label = y["_label_"].long()
            data_dict["label_top"].append(label.detach().cpu().numpy() / index_top)
            for var_name in var_names_to_save:
                data_dict[var_name].append(observers[var_name].detach().cpu().numpy())
            # calculate model predictions
            for model_name, d in models.items():
                model_output = d["model"](
                    inputs[0],
                    inputs[1][:, d["pf_features_indices"]],
                    inputs[2],
                    inputs[3],
                )
                # rescale the top quark prediction from ParT
                # we want p_Tbqq and p_QCD to add up to 1
                # --> p_top = p_Tbqq / (p_Tbqq + p_QCD)
                #     p_QCD = p_QCD  / (p_Tbqq + p_QCD)
                # --> p_top + p_QCD = 1
                p_SvsB = model_output[:, index_top] / (
                    model_output[:, index_top] + model_output[:, index_qcd]
                )
                data_dict[f"jet_p_top_{model_name}"].append(
                    p_SvsB.detach().cpu().numpy()
                )

    for key in data_dict:
        data_dict[key] = np.concatenate(data_dict[key])

    df = pd.DataFrame(data_dict)
    print(f"Saving dataset to {output_file}")
    df.to_hdf(output_file, key="df", mode="w")


def main():
    """Main function."""

    if DEVELOPMENT_MODE:
        print("Running in development mode. Only using small number of jets.")
        os.makedirs("./output_dev", exist_ok=True)

    # ----- Load pre-trained ParT models -----
    for model_name, model_cfg in models_to_evaluate.items():
        # Load the checkpoint from the pre-trained ParT model
        ckpt_pretrained = torch.load(model_cfg["model_path"], map_location="cuda")
        # Use the model configuration from
        # https://github.com/jet-universe/particle_transformer/blob/main/networks/example_ParticleTransformer.py#L26-L44  # noqa: E501
        model_kwargs = part_default_kwargs
        model_kwargs["input_dim"] = model_cfg["input_dim"]
        part_model_pretrained = ParticleTransformerWrapper(**model_kwargs)
        part_model_pretrained.load_state_dict(ckpt_pretrained)
        # Important: set the model to eval mode, otherwise the dropout layers will be
        # active and the model will perform worse
        part_model_pretrained.mod.for_inference = True
        part_model_pretrained.eval()
        part_model_pretrained.to("cuda")
        models_to_evaluate[model_name]["model"] = part_model_pretrained
        print(f"Loaded pre-trained ParT model from {model_cfg['model_path']}")

    input_names = ["pf_points", "pf_features", "pf_vectors", "pf_mask"]

    # Loop over train/val/test
    for ds_id in TRAIN_VAL_TEST_CONFIG:
        print(f"Processing {ds_id} dataset")
        # ----- Load JetClass data -----

        print("FILE_DICT:")
        for file_key, file_list in file_dict_jetclass[ds_id].items():
            print(file_key)
            for file in file_list:
                print(file)
            print("-----")

        dataset = SimpleIterDataset(
            file_dict=file_dict_jetclass[ds_id],
            data_config_file=DATA_CONFIG_JETCLASS,
            for_training=False,
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        os.makedirs("./output", exist_ok=True)
        output_folder = "./output_dev" if DEVELOPMENT_MODE else "./output"

        # ----- Create dataset with model predictions -----
        create_dataset_with_model_predictions(
            dataloader,
            models=models_to_evaluate,
            input_names=input_names,
            n_stop=TRAIN_VAL_TEST_CONFIG[ds_id]["n_filtered"],
            output_file=f"{output_folder}/filtered_jetclass_{ds_id}.h5",
            var_names_to_save=VAR_NAMES_TO_SAVE,
        )


if __name__ == "__main__":
    main()
