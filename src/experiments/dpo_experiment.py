import torch
from comet_ml import ExperimentConfig, start
from comet_ml.integration.pytorch import log_model, watch
from torch.utils.data import DataLoader

from src.datasets.dpo_dataset import DPODataset
from src.train.dpo_train import train_epoch


def dpo_experiment(model, ref_model, devices, api_key, name, split='train[:15%]'):
    experiment = start(
        api_key=api_key,
        project_name='NLP_HW4',
        workspace='soulqrat',
        mode='create',
        experiment_config=ExperimentConfig(
            name=name,
            log_code=False,
        ),
    )

    experiment.log_parameters(
        parameters={'model': name, 'lr': 1e-4, **model.params_cnt()}
    )

    model = model.to(devices[0])
    ref_model = ref_model.to(devices[1])
    dataset = DPODataset(model.tokenizer, split=split)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(
        [params for params in model.parameters() if params.requires_grad], lr=1e-4
    )

    train_epoch(
        model,
        ref_model,
        dataloader,
        optimizer,
        epoch=0,
        n_epochs=1,
        devices=devices,
        scheduler=None,
        experiment=experiment,
    )

    log_model(experiment, model, name)
