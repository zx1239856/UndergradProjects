from sklearn.metrics import accuracy_score, f1_score
import torch


class AverageMeter(object):
    def __init__(self, total=0.0, count=0):
        self.total = total
        self.count = count

    def update(self, value):
        self.total += value
        self.count += 1

    def mean(self):
        return self.total / self.count if self.count != 0 else 0


class Metrics(object):
    def __init__(self, loss=0, cls_loss=0, ctx_loss=0, cls_acc=0, ctx_acc=0, cls_f1=0, ctx_f1=0):
        self.loss = loss
        self.cls_loss = cls_loss
        self.ctx_loss = ctx_loss
        self.cls_acc = cls_acc
        self.ctx_acc = ctx_acc
        self.cls_f1 = cls_f1
        self.ctx_f1 = ctx_f1

    def __str__(self):
        return (f'Metrics(loss={self.loss:.4f}, cls_loss={self.cls_loss:.4f}, ctx_loss={self.ctx_loss:.4f}, '
                f'cls_acc={self.cls_acc:.4f}, ctx_acc={self.ctx_acc:.4f}, '
                f'cls_f1={self.cls_f1:.4f}, ctx_f1={self.ctx_f1:.4f})')

    def __repr__(self):
        return self.__str__()


class MetricsAverageMeter(object):
    def __init__(self):
        for key in Metrics().__dict__.keys():
            setattr(self, key, AverageMeter())

    def update(self, metrics):
        for key, value in self.__dict__.items():
            value.update(getattr(metrics, key))

    def mean(self):
        metrics = Metrics()
        for key, value in self.__dict__.items():
            setattr(metrics, key, value.mean())
        return metrics


def log_metrics(writer, global_step, stage, metrics):
    writer.add_scalar(f'{stage}/loss', metrics.loss.data.item(), global_step=global_step)
    writer.add_scalar(f'{stage}/cls_loss', metrics.cls_loss.data.item(), global_step=global_step)
    writer.add_scalar(f'{stage}/ctx_loss', metrics.ctx_loss.data.item(), global_step=global_step)
    writer.add_scalar(f'{stage}/cls_acc', metrics.cls_acc, global_step=global_step)
    writer.add_scalar(f'{stage}/ctx_acc', metrics.ctx_acc, global_step=global_step)
    writer.add_scalar(f'{stage}/cls_f1', metrics.cls_f1, global_step=global_step)
    writer.add_scalar(f'{stage}/ctx_f1', metrics.ctx_f1, global_step=global_step)


def get_metrics(cls_pred, ctx_pred, cls_labels, ctx_labels, criterion_cls, criterion_ctx, ctx_weight):
    stats = Metrics()

    stats.cls_loss = criterion_cls(cls_pred, cls_labels)
    stats.ctx_loss = criterion_ctx(ctx_pred, ctx_labels)
    stats.loss = stats.cls_loss + ctx_weight * stats.ctx_loss

    cls_pred_labels = cls_pred.argmax(dim=-1).cpu()
    ctx_pred_labels = ctx_pred.argmax(dim=-1).cpu()
    cls_labels = cls_labels.cpu()
    ctx_labels = ctx_labels.cpu()

    stats.cls_acc = accuracy_score(cls_pred_labels, cls_labels)
    stats.ctx_acc = accuracy_score(ctx_pred_labels, ctx_labels)
    stats.cls_f1 = f1_score(cls_pred_labels, cls_labels, average='macro')
    stats.ctx_f1 = f1_score(ctx_pred_labels, ctx_labels, average='macro')

    return stats
