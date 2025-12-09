import torch
import torch.optim as optim


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler
