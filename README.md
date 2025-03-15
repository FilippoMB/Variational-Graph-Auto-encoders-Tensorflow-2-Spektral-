# Variational Graph Auto-encoders in Spektral

This is a re-implementation of the Graph Auto-encoder and Variational Graph Auto-encoder presented [here](https://arxiv.org/pdf/1611.07308.pdf).

This implementation is based on [Spektral](https://graphneural.network/), the Tensorflow-Keras library for Graph Neural Networks.
Compared to the [original implementation](https://github.com/tkipf/gae), the code is more compact and the scripts are self-contained.

The results obtained are slightly different from those reported in the original paper (the AUC is actually higher here). 
Also, the results show that there is not a significant improvement in using the variational variant compared to the deterministic one.
