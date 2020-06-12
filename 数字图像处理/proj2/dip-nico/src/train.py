import json
import logging
import os

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import init_helper
import datasets
import models
import metrics
import evaluate

logger = logging.getLogger()


def main():
    args = init_helper.get_arguments()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, 'checkpoints'), exist_ok=True)

    init_helper.init_logger(os.path.join(args.model_dir, 'logs'), 'train.log')
    init_helper.set_random_seed(args.seed)

    with open(os.path.join(args.model_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, separators=(',', ': '))

    # model = models.Classifier()
    model = models.AttentionClassifier()

    logger.info(args)

    optimizer = Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_sched = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    global_step = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_dict'])
        if args.resume_step:
            global_step = checkpoint['global_step'] + 1

    train_set = datasets.NicoPairedDataset(args.train_split, args.device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = datasets.NicoDataset(args.val_split, args.device)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f'Train-set {len(train_set)}. Val-set {len(val_set)}.')

    criterion_cls = nn.CrossEntropyLoss().to(args.device)
    criterion_ctx = nn.BCEWithLogitsLoss().to(args.device)

    model.to(args.device)

    with SummaryWriter(os.path.join(args.model_dir, 'board')) as writer:

        for epoch in range(args.max_epoch):
            logger.info(f'Epoch {epoch} (step {global_step}) started')

            stats_meter = metrics.MetricsAverageMeter()

            model.train()

            for batch_index, ((feat_src, cls_src, ctx_src), (feat_tgt, cls_tgt, ctx_tgt)) in enumerate(train_loader):
                x = torch.cat([feat_src, feat_tgt])
                feats = model.attention(x)
                ctx_pred = model.discriminator(feats.view(feat_src.shape[0], -1))
                cls_pred = model.fc_cls(feats)
                ctx_labels = (ctx_src == ctx_tgt).view((-1, 1)).type(torch.float32)
                cls_labels = torch.cat([cls_src, cls_tgt])

                stats = metrics.get_metrics(cls_pred, ctx_pred, cls_labels, ctx_labels, criterion_cls, criterion_ctx, args.ctx_weight)

                # back prop
                optimizer.zero_grad()
                stats.loss.backward()
                optimizer.step()

                stats_meter.update(stats)
                metrics.log_metrics(writer, global_step, 'train', stats)

                # log learning rate
                if global_step % args.lr_decay_step == 0:
                    for idx, param_group in enumerate(optimizer.param_groups):
                        group_lr = param_group['lr']
                        logger.info(f'Learning rate of group {idx} decayed to {group_lr}')
                        writer.add_scalar(f'lr/group_{idx}', group_lr, global_step=global_step)

                lr_sched.step()
                global_step += 1

            logger.info(f'(Train) {stats_meter.mean()}')

            # save checkpoints
            torch.save({
                'model_dict': model.state_dict(),
                'global_step': global_step
            }, os.path.join(args.model_dir, 'checkpoints', f'{global_step}.pt'))

            # evaluate after every epoch
            stats = evaluate.evaluate(model, val_loader, criterion_cls, args.ctx_weight, args.device)
            metrics.log_metrics(writer, global_step, 'val', stats)

            logger.info(f'(Val) {stats}')


if __name__ == '__main__':
    main()
