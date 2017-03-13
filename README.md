# mnist-arch
### A couple of weird and apparently pointless MNIST architectures

 * The LSTM one poses the MNIST classification problem as an image being a 'sequence of pixel lines problem'
 * The MLP one is just a pile of dumb fully connected neurons arranged weirdly in an arrow / trapezoid shape, it's there mainly for comparison
 * Then there's an architecture that's merging the results of the previous ones by summin/concat etc. in an attempt to see if there's any value in doing it

[![MLP](https://raw.githubusercontent.com/adaptorel/mnist-arch/master/_graphs/mlp_mnist.png)](https://raw.githubusercontent.com/adaptorel/mnist-arch/master/_graphs/mlp_mnist.png)
[![LSTM](https://raw.githubusercontent.com/adaptorel/mnist-arch/master/_graphs/lstm_mnist.png)](https://raw.githubusercontent.com/adaptorel/mnist-arch/master/_graphs/lstm_mnist.png)
[![MLP w/ LSTM](https://raw.githubusercontent.com/adaptorel/mnist-arch/master/_graphs/mlp_w__lstm_mnist.png)](https://raw.githubusercontent.com/adaptorel/mnist-arch/master/_graphs/mlp_w__lstm_mnist.png)
