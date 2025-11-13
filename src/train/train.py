import time

import torch
from tqdm import tqdm


def train_epoch(
    model,
    dataloader,
    optimizer,
    epoch,
    n_epochs,
    device,
    scheduler=None,
    experiment=None,
    collect_stats=True,
):
    model.train()
    total_loss = 0
    total_t_forward = 0
    total_t_backward = 0
    for i, (data, mask) in enumerate(
        tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
    ):
        data = data.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()

        if collect_stats and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t_forward = time.perf_counter()
        outputs = model(input_ids=data, attention_mask=mask, labels=data)
        t_forward = time.perf_counter() - t_forward

        if collect_stats and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated()

        loss = outputs.loss
        t_backward = time.perf_counter()
        loss.backward()
        t_backward = time.perf_counter() - t_backward

        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch + i / len(dataloader))

        total_loss += loss.item()
        total_t_forward += t_forward
        total_t_backward += t_backward
        if experiment is not None:
            experiment.log_metric("loss", loss.item(), step=epoch * len(dataloader) + i)
            if collect_stats:
                experiment.log_metric(
                    "forward_time", t_forward, step=epoch * len(dataloader) + i
                )
                experiment.log_metric(
                    "backward_time", t_backward, step=epoch * len(dataloader) + i
                )
                if torch.cuda.is_available():
                    experiment.log_metric(
                        "memory", memory, step=epoch * len(dataloader) + i
                    )

    if experiment is not None:
        experiment.log_metric(
            "epoch_loss", total_loss / len(dataloader), step=epoch + 1
        )
        if collect_stats:
            experiment.log_metric(
                "epoch_forward_time", total_t_forward / len(dataloader), step=epoch + 1
            )
            experiment.log_metric(
                "epoch_backward_time",
                total_t_backward / len(dataloader),
                step=epoch + 1,
            )
