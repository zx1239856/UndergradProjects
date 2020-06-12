import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--train-set', type=str, default='../data/course_train.npy')
    parser.add_argument('--test-set', type=str, default='../data/test_data.npy')
    parser.add_argument('--train-context', type=int, default=6)
    parser.add_argument('--save-dir', type=str, default='../split/')
    args = parser.parse_args()

    np.random.seed(args.seed)

    num_cls = 10
    num_ctx = 7

    data = np.load(args.train_set)
    ctx_labels = data[:, -2].astype(np.int32)
    cls_labels = data[:, -1].astype(np.int32)

    train_mask = np.zeros(ctx_labels.shape, dtype=np.bool)
    # select train contexts for each class
    for clazz in range(num_cls):
        cls_ctx_labels = np.unique(ctx_labels[cls_labels == clazz])
        assert len(cls_ctx_labels) == num_ctx

        train_ctx = cls_ctx_labels[:args.train_context]

        for idx, (cls_label, ctx_label) in enumerate(zip(cls_labels, ctx_labels)):
            if cls_label == clazz and ctx_label in train_ctx:
                train_mask[idx] = True

    train_set = data[train_mask]
    val_set = data[~train_mask]

    print(f'Train {np.sum(train_mask)} samples')
    print(f'Val {np.sum(~train_mask)} samples')

    # validate
    assert train_set.size + val_set.size == data.size
    for clazz in range(num_cls):
        cls_mask = train_set[:, -1].astype(np.int32) == clazz
        assert len(set(train_set[cls_mask, -2].astype(np.int32))) == args.train_context
        cls_mask = val_set[:, -1].astype(np.int32) == clazz
        assert len(set(val_set[cls_mask, -2].astype(np.int32))) == num_ctx - args.train_context

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'train.npy'), train_set)
    np.save(os.path.join(args.save_dir, 'val.npy'), val_set)

    # add dummy class labels in test data
    data = np.load(args.test_set)
    num_test, _ = data.shape
    cls_labels = np.full((num_test, 1), 0, dtype=data.dtype)
    test_set = np.hstack((data, cls_labels))
    np.save(os.path.join(args.save_dir, 'test.npy'), test_set)
    print(f'Test {num_test} samples')


if __name__ == '__main__':
    main()
