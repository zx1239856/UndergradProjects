# NICO Classifier

## Getting Started

Create a virtual env with python 3.6 (optional).

```sh
virtualenv -p /usr/bin/python3.6 venv36
. venv36/bin/activate
```

Install dependencies.

```sh
pip install -r requirements.txt
```

## Dataset Preparation

Place the train dataset `course_train.npy` and test dataset `test_data.npy` in `data/` folder.

Split the training dataset into train-set and val-set, and add dummy class labels in test data.

```sh
python split.py
```

Now your project structure should look like this.

```
dip-nico/
├── data
│   ├── course_train.npy
│   └── test_data.npy
├── README.md
├── split
│   ├── test.npy
│   ├── train.npy
│   └── val.npy
└── src
    └── xxx.py
```

## Training

Train

```sh
python train.py
```

## Evaluation

Evaluate on validation set

```sh
python evaluate.py --resume ../models/model/checkpoints/xxx.pt
```

Evaluate on test set

```sh
python evaluate.py --val-split ../split/test.npy \
    --resume ../models/model/checkpoints/xxx.pt --save-dir ../models/model/results/xxx/
```
