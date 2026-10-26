# Examples

Scripts in the `examples/` folder of the repository. Run `python <script> --help` for all options.

## Basic

### xor.py

Train discrete or guided TM on noisy XOR (parity of --nbits relevant bits, plus --noisybits random bits).

```{literalinclude} ../examples/xor.py
:language: python
```

### iris.py

Train discrete or guided TM on Iris (4 numeric features, 3 classes).

```{literalinclude} ../examples/iris.py
:language: python
```

## Image

### image/mnist.py

Train discrete or guided TM on MNIST dataset.

```{literalinclude} ../examples/image/mnist.py
:language: python
```

### image/fmnist.py

Train discrete or guided TM on Fashion-MNIST dataset.

```{literalinclude} ../examples/image/fmnist.py
:language: python
```

### image/kmnist.py

Train discrete or guided TM on Kuzushiji-MNIST dataset.

```{literalinclude} ../examples/image/kmnist.py
:language: python
```

### image/cifar10.py

Train discrete or guided TM on CIFAR-10 dataset.

```{literalinclude} ../examples/image/cifar10.py
:language: python
```

### image/celeba.py

Train discrete or guided TM on CelebA dataset (40 binary face attributes, multilabel).

```{literalinclude} ../examples/image/celeba.py
:language: python
```

## NLP

### nlp/imdb.py

Train discrete or guided TM on IMDB movie reviews (binary sentiment).

```{literalinclude} ../examples/nlp/imdb.py
:language: python
```

### nlp/ag_news.py

Train discrete or guided TM on AG News (4 news topics).

```{literalinclude} ../examples/nlp/ag_news.py
:language: python
```

### nlp/go_emotions.py

Train discrete or guided TM on GoEmotions (28 emotion labels, multilabel).

```{literalinclude} ../examples/nlp/go_emotions.py
:language: python
```

## Regression

### reg/friedman1.py

Train discrete or guided regression TM on Friedman #1 (synthetic, sklearn make_friedman1).

```{literalinclude} ../examples/reg/friedman1.py
:language: python
```

### reg/diabetes.py

Train discrete or guided regression TM on Diabetes (442 patients, disease progression after one year).

```{literalinclude} ../examples/reg/diabetes.py
:language: python
```

### reg/california_housing.py

Train discrete or guided regression TM on California Housing (8 numeric features, median house value in $100k).

```{literalinclude} ../examples/reg/california_housing.py
:language: python
```
