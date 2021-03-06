# Variational Graph Auto-encoders (Tensorflow 2 + Spektral)

This is a re-implementation of the Graph Auto-encoder and Variational Graph Auto-encoder presented [here](https://arxiv.org/pdf/1611.07308.pdf).

This implementation is based on Tensorflow 2 and [Spektral](https://graphneural.network/).
Compared to the [original implementation](https://github.com/tkipf/gae), the code is more compact and the scripts are self-contained.

The results obtained are slightly different from those reported in the original paper (the AUC is higher). Also, there is not a clear improvement in using the variational version compared to the determistic one.
