import torch
from comet_ml import ExperimentConfig, start
from comet_ml.integration.pytorch import log_model, watch
from torch.utils.data import DataLoader

from src.datasets.rlhf_dataset import RLHFDataset
from src.train.train import train_epoch_labels


def pythia_experiment(model, device, api_key, name):
    experiment = start(
        api_key=api_key,
        project_name="NLP_HW4",
        workspace="soulqrat",
        mode="create",
        experiment_config=ExperimentConfig(
            name=name,
            log_code=False,
        ),
    )

    experiment.log_parameters(
        parameters={"model": name, "lr": 1e-4, **model.params_cnt()}
    )

    model = model.to(device)

    dataset = RLHFDataset(model.tokenizer, split="train")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(
        [params for params in model.parameters() if params.requires_grad], lr=1e-4
    )

    train_epoch_labels(
        model,
        dataloader,
        optimizer,
        epoch=0,
        n_epochs=1,
        device=device,
        scheduler=None,
        experiment=experiment,
    )

    log_model(experiment, model, name)
