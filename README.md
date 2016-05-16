## TensorFlow implementation of "Iterative Pruning"

This work is based on "Learning both Weights and Connections for Efficient
Neural Network." [Song et al.](http://arxiv.org/pdf/1506.02626v3.pdf) @ NIPS '15.
Note that these works are just for quantifying its effectiveness on latency (within TensorFlow),
not a best optimal. Thus, some details are abbreviated for simplicity. (e.g. # of iterations, adjusted dropout ratio, etc.)

I applied Iterative Pruning on a small MNIST CNN model (13MB, originally), which can be
accessed from [TensorFlow Tutorials](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html).
After pruning off some percentages of weights, I've simply retrained two epochs for
each case and got compressed models (minimum 2.6MB with 90% off) with minor loss of accuracy.
(99.17% -> 98.99% with 90% off and retraining) Again, this is not an optimal.

Due to lack of supports on SparseTensor and its operations of TensorFlow (0.8.0),
this implementation uses [*embedding_lookup_sparse*](https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#embedding_lookup_sparse)
rather than using sparse matmul that uses typical CSR format, such as cuSPARSE.
Also, I think this implementation has following limitations.

1. *embedding_lookup_sparse* doesn't support ```broadcasting```, which prohibits users to run test with normal test datasets.
2. Performance may be somewhat sub-optimal.
3. Because Sparse Variable is not supported, manual dense to sparse and sparse to dense transformation is required.
4. 4D Convolution Tensor may also be applicable, but bit tricky.
5. *embedding_lookup_sparse* forces additional matrix transpose, dimension squeeze and dimension reshape.

## File descriptions and usages

model_ckpt_dense: original model<br>
model_ckpt_dense_pruned: 90% pruned-only model<br>
model_ckpt_sparse_retrained: 90% pruned and retrained model<br>

To regenerate these sparse model, edit ```config.py``` first as your threshold configuration,
and then run training with second (pruning and retraining) and third (generate sparse form of weight data) round options.

```bash
./train.py -2 -3
```

To inference single image (seven.png) and measuring its latency,

```bash
./deploy_test.py -d -m model_ckpt_dense
./deploy_test_sparse.py -d -m model_ckpt_sparse_retrained
```

To test dense model,

```bash
./deploy_test.py -t -m model_ckpt_dense
./deploy_test.py -t -m model_ckpt_dense_pruned
./deploy_test.py -t -m model_ckpt_dense_retrained
```

To draw histogram that shows the weight distribution,

```bash
# After running train.py (it generates .dat files)
./draw_histogram.py
```

## Performance
Results are currently somewhat mediocre or degraded due to indirection and additional storage overhead originated from sparse matrix form.
Also, it may because model size is too small. (12.49MB)

#### Storage overhead
Baseline: 12.49 MB<br>
10 % pruned: 21.86 MB<br>
20 % pruned: 19.45 MB<br>
30 % pruned: 17.05 MB<br>
40 % pruned: 14.64 MB<br>
50 % pruned: 12.23 MB<br>
60 % pruned: 9.83 MB<br>
70 % pruned: 7.42 MB<br>
80 % pruned: 5.02 MB<br>
90 % pruned: 2.61 MB<br>

#### CPU performance (5 times averaged)
CPU: Intel Core i5-2500 @ 3.3 GHz,
LLC size: 6 MB

<img src=http://garion9013.github.io/images/cpu-desktop.png alt=http://garion9013.github.io/images/cpu-desktop.png>

Baseline: 0.01118040085 s<br>
10 % pruned: 0.005825249279 s<br>
20 % pruned: 0.04808278927  s<br>
30 % pruned: 0.05296058167  s<br>
40 % pruned: 0.05639345492  s<br>
50 % pruned: 0.06608673888  s<br>
60 % pruned: 0.08565862006  s<br>
70 % pruned: 0.09634974708  s<br>
80 % pruned: 0.1079562668   s<br>
90 % pruned: 0.1056580001   s<br>

#### GPU performance (5 times averaged)
GPU: Nvidia Geforce GTX650 @ 1.058 GHz,
LLC size: 256 KB

<img src=http://garion9013.github.io/images/gpu-desktop.png alt=http://garion9013.github.io/images/gpu-desktop.png>

Baseline: 0.1475181845 s<br>
10 % pruned: 0.4992931957 s<br>
20 % pruned: 0.5534564735 s<br>
30 % pruned: 0.5705291375 s<br>
40 % pruned: 0.7058111907 s<br>
50 % pruned: 0.7393358328 s<br>
60 % pruned: 0.8126856559 s<br>
70 % pruned: 1.02670899   s<br>
80 % pruned: 1.08733898   s<br>
90 % pruned: 1.210453449  s<br>
