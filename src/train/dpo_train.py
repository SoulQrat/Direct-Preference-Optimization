from tqdm import tqdm
import torch
from src.loss.dpo import DPOLoss

def train_epoch(model,
    ref_model,
    dataloader,
    optimizer,
    epoch,
    n_epochs,
    devices,
    scheduler=None,
    experiment=None,
    ):
    
    model.train()
    ref_model.eval()
    total_loss = 0

    for i, (data, mask, labels_chosen, labels_rejected) in enumerate(
        tqdm(dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')
    ):
        data = data.to(devices[0])
        mask = mask.to(devices[0])
        labels_chosen = labels_chosen.to(devices[0])
        labels_rejected = labels_rejected.to(devices[0])

        optimizer.zero_grad()

        out_model_w = model(input_ids=data, attention_mask=mask, labels=labels_chosen)
        out_model_l = model(input_ids=data, attention_mask=mask, labels=labels_rejected)

        data = data.to(devices[1])
        mask = mask.to(devices[1])
        labels_chosen = labels_chosen.to(devices[1])
        labels_rejected = labels_rejected.to(devices[1])
        with torch.no_grad():
            out_ref_model_w = ref_model(input_ids=data, attention_mask=mask, labels=labels_chosen)
            out_ref_model_l = ref_model(input_ids=data, attention_mask=mask, labels=labels_rejected)

        loss = DPOLoss(out_model_w, out_model_l, out_ref_model_w, out_ref_model_l, mask, labels_chosen, labels_rejected, beta=0.5)
        
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch + i / len(dataloader))

        total_loss += loss.item()
        if experiment is not None:
            experiment.log_metric('loss', loss.item(), step=epoch * len(dataloader) + i)
    
    if experiment is not None:
        experiment.log_metric(
            'epoch_loss', total_loss / len(dataloader), step=epoch + 1
        )