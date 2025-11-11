from src.train.train import train_epoch
from src.datasets.text_dataset import TextDataset

from comet_ml import start, ExperimentConfig
from comet_ml.integration.pytorch import log_model, watch

import torch
from torch.utils.data import DataLoader

def time_experiment(model, device, api_key, name):
    experiment = start(
        api_key = api_key,
        project_name="NLP_HW4",
        workspace="soulqrat",
        mode="create",
        experiment_config=ExperimentConfig(
                name=name,
                log_code=False,
            ),
    )

    experiment.log_parameters(parameters={'model': name, 'lr': 1e-4, **model.params_cnt()})

    model = model.to(device)

    dataset = TextDataset(model.tokenizer, split='train[:10%]')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW([params for params in model.parameters() if params.requires_grad], lr=1e-4)


    train_epoch(model, dataloader, optimizer, epoch=0, n_epochs=1, device=device, scheduler=None, experiment=experiment)