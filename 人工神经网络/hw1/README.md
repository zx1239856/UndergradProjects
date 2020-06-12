# Multi-layer Perceptron Implementation

## Changelog

+ Forward and back propagation, as well as loss functions are properly implemented
+ Rewrite `run_mlp.py` to support constructing network from command line arguments
+ Add some changes to `train_net` in `solve_net.py`, in order to get accuracy and loss arrays during iterations.
+ Support debug in `layers.py` (can be enabled via `setDebug`). Once set to `True`, the L2-norm of gradient will output during the BP-phase of training.

## Usage

```bash
python3 run_mlp.py --help  ## use this line to see how to use
```

## How to Reproduce Results in the Report

```bash
python3 run_mlp.py -layer $layer_structure -e 50 -loss mse -out output_filename ## set layer structure accordingly, see Appendix A of the report
```

### Loss functions

Available options are: `mse` and `cross_entropy`

### Layer Structure

This is a string parameter, where you describe the structure in a linear fashion (Similar to `Sequential` model in `keras`). Available options are

+ linear, layer_name, input_dim, out_dim, initial_stdev
+ sigmoid, layer_name
+ relu, layer_name

Different layers should be separated by a single semicolon `;`

### Example

Say, if you want a linear layer with 500 inputs and 300 outputs, followed by a Sigmoid activation, then a linear layer with 300 inputs and 10 outputs, followed by a ReLU activation, then the `$layer_structure` should be

```
linear,fc1,500,300;sigmoid,sg;linear,fc2,300,10;relu,relu
```

Spaces between tokens(linear, sigmoid, punctuations, layer_name and things like these) are allowed.

## Requirements

+ python >= 3.6
+ numpy >= 1.12.0