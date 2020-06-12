import logging
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import init_helper
import metrics
import models
import datasets

logger = logging.getLogger()


def evaluate(model, dataloader, criterion, ctx_weight, device, save_dir=None):
    model = model.to(device).eval()

    stats_meter = metrics.MetricsAverageMeter()

    cls_preds = []

    with torch.no_grad():

        for batch_index, (features, cls_labels, ctx_labels) in enumerate(dataloader):
            cls_pred, ctx_pred = model(features)

            stats = metrics.get_metrics(cls_pred, ctx_pred, cls_labels, ctx_labels, criterion, criterion, ctx_weight)
            stats_meter.update(stats)

            cls_preds += cls_pred.argmax(dim=-1).tolist()

    if save_dir is not None:
        # save class prediction only
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'submit.txt'), 'w') as f:
            f.write('\n'.join([str(x) for x in cls_preds]))

    stats = stats_meter.mean()
    return stats


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(os.path.join(args.model_dir, 'logs'), f'{os.path.basename(args.resume)}.log')
    init_helper.set_random_seed(args.seed)

    checkpoint = torch.load(args.resume, map_location=lambda storage, location: storage)
    model_dict = checkpoint['model_dict']

    # model = models.Classifier()
    model = models.AttentionClassifier()
    model = model.to(args.device).eval()
    model.load_state_dict(model_dict)

    logger.info(args)

    dataset = datasets.NicoDataset(args.val_split, args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss().to(args.device)
    stats = evaluate(model, loader, criterion, args.ctx_weight, args.device, args.save_dir)
    logger.info(f'(Val) {stats}')


if __name__ == '__main__':
    main()
