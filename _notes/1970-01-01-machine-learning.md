---
layout: note
title: Machine Learning
description: High-level overview of the incredibly large field of Machine Learning.
authors: [tlyleung]
x: 50
y: 40
---

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/neural-network.svg width="100%" height="100%" %}</div>
# Machine Learning

This cheatsheet attempts to give a high-level overview of the incredibly large field of Machine Learning. Please [contact me](https://x.com/tlyleung) for corrections/omissions.

_Last updated: 20 August 2025_

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/book-flip-page.svg width="100%" height="100%" %}</div>
# Contents

- [Background](#background)
- [Machine Learning Lifecycle](#machine-learning-lifecycle)
- [Problem Framing](#problem-framing)
- [Data Assembly](#data-assembly)
- [Model Training](#model-training)
  - [PyTorch](#model-training-pytorch)
- [Model Evaluation](#model-evaluation)
  - [Responsible AI](#model-evaluation-responsible-ai)
  - [Metrics](#model-evaluation-metrics)
- [Model Deployment](#model-deployment)
- [Models](#models)
  - [Supervised Learning](#models-supervised-learning)
  - [Unsupervised Learning](#models-unsupervised-learning)
  - [Reinforcement Learning](#models-reinforcement-learning)
  - [Recommender Systems](#models-recommender-systems)
  - [Ensembles](#models-ensembles)
  - [Tasks](#models-tasks)
- [Designs](#designs)
  - [Visual Search](#designs-visual-search)
  - [Google Street View Blurring](#designs-google-street-view-blurring)
  - [YouTube Video Search](#designs-youtube-video-search)
  - [Harmful Content Detection](#designs-harmful-content-detection)
  - [Video Recommendation System](#designs-video-recommendation-system)
  - [Event Recommendation System](#designs-event-recommendation-system)
  - [Ad Click Prediction on Social Platforms](#designs-ad-click-prediction-on-social-platforms)
  - [Similar Listings on Vacation Rental Platforms](#designs-similar-listings-on-vacation-rental-platforms)
  - [Personalized News Feed](#designs-personalized-news-feed)
  - [People You May Know](#designs-people-you-may-know)
</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/official-building-3.svg width="100%" height="100%" %}</div>
# Background

- Artificial Intelligence is the ability of a machine to demonstrate human-like intelligence.
- Machine Learning is the field of study that gives computers the ability to learn without explicitly being programmed.
- Machine Learning has become possible because of:
  - Massive labelled datasets, e.g. ImageNet
  - Improved hardware and compute, e.g. GPUs
  - Algorithms advancements, e.g. backpropagation

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/synchronize-arrows-three.svg width="100%" height="100%" %}</div>

# Machine Learning Lifecycle

1. Problem Framing
2. Data Assembly
3. Model Training
4. Model Evaluation
5. Model Deployment

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/bulb-1.svg width="100%" height="100%" %}</div>
# Problem Framing

### Establish Business Objective

- What does success looks like from a business perspective?
- What action is taken as a result of this prediction?
- How does this action impact the business?

### Determine Core System Features (Functional Requirements)

- Users should be able to do X?
- Does the system need to do Y?
- What would happen if Z?

### Determine System Qualities (Non-functional Requirements)

- The system should be able to... (include quantity)
- The system should be... (include quantity)
- See [Non-functional Requirements](#non-functional-requirements) for more details.

### Define Product (Online) Metric

- Did this system create business or user value?
- **Engagement:** click-through-rate, engagement rate, completion rate
- **Business:** conversion rate, revenue per user, churn rate
- **User experience:** user satisfaction, user retention
- **System performance:** latency

### Define ML (Offline) Metric

- Use relevant classification, regression, ranking or generation metrics.

### Sketch High-level Design

- Draw a simple block diagram showing the inputs/outputs of the system together with the key components in between

---

## When to use Machine Learning?[^huyen22]

Machine Learning is an approach to <u>learn</u> <u>complex patterns</u> from <u>existing data</u> and use these <u>patterns</u> to make <u>predictions</u> on <u>unseen data</u>.

- **Learn:** the system has the capacity to learn.
- **Complex patterns:** there are patterns to learn and they are complex.
- **Existing data:** data is available or it's possible to collect data.
- **Predictions:** it's a predictive problem.
- **Unseen data:** unseen data shares patterns with the training data.

Machine Learning systems work best when:

- Data is repetitive
- Cost of wrong predictions is cheap
- Problem is at scale
- Patterns are constantly changing

---

## Non-functional Requirements[^cs329s][^mlsystemdesign]

### Performance

- Cost of wrong predictions
- False negatives vs. false positives
- Interpretability
- Usefulness threshold

### Training

- Freshness requirements
- Training frequency

### Inference

- Computing power
- Confidence measurement (if confidence is below threshold: discard, clarify or refer to humans?)
- Cost
- Number of items
- Number of users
- Latency
- Peak requests

### Online vs. Batch

- **Online:** generate predictions as requests arrive, e.g. speech recognition
- **Batch:** generate predictions periodically before requests arrive, e.g. Netflix recommendations

### Cloud vs. Edge vs. Hybrid

- **Cloud:** no energy, power or memory constraints
- **Edge:** can work without unreliable connections, no network latency, fewer privacy concerns, cheaper
- **Hybrid:** common predictions are precomputed and stored on device

### Privacy

- **Annotations:** can data be shipped to outside organisations?
- **Storage:** what data are you allowed to store and for how long?
- **Third-party:** can you share data with a third-party?
- **Regulations:** is the data complying with relevant data protection laws?

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/database-2.svg width="100%" height="100%" %}</div>
# Data Assembly

1. **Data Collection:** (see below)
2. **Exploratory Data Analysis (EDA):** use visualisation and statistical techniques to understand the data's structure, detect patterns and spot anomalies
3. **Data Preprocessing:** (see below)
4. **Feature Engineering:** (see below)
5. **Feature Selection:** remove features with low variance, recursive feature elimination, sequential feature selection
6. **Sampling Strategy:** sequential, random, stratified, weighted, reservoir, importance
7. **Data Splits:** train-test-validation split, windows splitting of time series data

---

## Class Imbalance[^cs329s]

### Collect More Data

- Target under-represented classes

### Data-level Methods

- Undersample majority class (can cause overfitting), e.g. Tomek Links makes decision boundaries clearer by finding pairs of close samples from opposite classes and removes the majority sample
- Oversample minority class (can cause loss of information), e.g. generate synthetic minority oversampling (SMOTE)

### Algorithm-level Methods

- Cost-sensitive learning penalises the misclassification of minority class samples more heavily than majority class samples
- Class-balance loss by giving more weight to rare classes
- Focal loss by giving more weight to difficult samples

---

## Data Collection[^data_checklist]

Good data should:

- have good predictive power (an expert should be able to make a correct prediction with the data)
- have very little missing values (when missing values do occur, they should be explainable and occur randomly)
- be labelled
- be correct and accurate
- be documented
- be unbiased

### Data Biases[^cs329s]

- Sampling/selection bias
- Under/over representation of subgroups
- Human biases embedded in the data
- Labelling bias
- Algorithmic bias

### Data Labelling[^cs329s]

- Hand-labelling, data lineage (track where data/labels come from)
- Use Labelling Functions (LFs) to label training data programmatic using different heuristics, including pattern matching, boolean search, database lookup, prediction from legacy system, third-party model, crowd labels
- Weak supervision, semi supervision, active learning, transfer learning

---

## Data Leakage[^cs329s]

- Splitting time-correlated data randomly instead of by time
- Preprocessing data before splitting, e.g. using the whole dataset to generate global statistics like the mean and using it to impute missing values
- Poorly handling of data duplication before splitting
- Group leakage, group of examples have strongly correlated labels but are divided into different splits
- Leakage from data collection process, e.g. doctors sending high-risk patients to a better scanner
- Detect data leakage by measuring correlation of a feature with labels, feature ablation study, monitoring model performance when new features are added

---

## Data Preprocessing[^sklearn]

### Missing Data

- Collect more data
- Drop row/column
- Constant imputation
- Univariate imputation: replace missing values with the column mean/median/mode
- Multivariate imputation: use all available features to estimate missing values
- Nearest neighbours imputation: use an euclidean distance metric to find nearest neighbors
- Add missing indicator column

### Missing Values

- **Missing Not At Random (MNAR):** missing due to the value itself
- **Missing At Random (MAR):** missing due to another observed variable
- **Missing Completely At Random (MCAR):** no pattern to which values are missing

### Structured Data

- **Categorical:** ordinal encoding, one-hot encoding
- **Numeric:** discretisation, linear scaling (when the feature is uniformly distributed across the range), z-score normalisation (when the feature is normally distributed), log scaling (when the feature distribution is heavily skewed on one side), clipping (when the feature contains extreme outliers), power transform (mapping to Gaussian distribution using Yeo-Johnson or Box-Cox transforms)

### Unstructured Data

- **Audio:** sampling, noise reduction, normalisation, feature extraction, silence removal
- **Images:** decoding, resizing, normalisation, augmentation
- **Text:** normalisation (lower-casing, punctuation removal, strip whitespaces, strip accents, lemmatisation and stemming), tokenisation, token to IDs, stopword removal
- **Videos:** frame extraction, resizing, normalisation, optical flow analysis, augmentation

### Mask Personal Data

- **Personal Identifiable Information (PII):** name, email, phone number, address, date of birth, etc.
- **Personal Data:** device ID, IP address, browser fingerprint, etc.
- **Sensitive Data:** philosophical beliefs, trade-union membership, genetic data, medical data, religious beliefs, sexual orientation, etc.

---

## Feature Engineering[^sklearn]

Use domain knowledge to extract and transform predictive features from raw data into a format usable by the model.

- **Dimensionality Reduction:** use Principal Component Analysis (PCA) to find a subset of features that capture the variance of the original features or Hierarchical Clustering to group features that behave similarly.
- **Feature Crossing:** combine two or more features to create a new feature.
- **Positional Embeddings:** can be either learned or fixed at test time.

### Example Features (Event Recommendation System)

- **Events:** type, price
- **Location:** walk score, transit score, same region, distance to event, venue capacity,
- **Social:** total attendance, friend attendance, invited by other user, hosted by a friend, attendance of events by same host, social media engagement
- **Time:** remaining time until event begins, estimated travel time, time of day, weekday vs. weekend, seasonality, frequency, duration
- **User:** age, gender, past attendance, event preferences, income bracket

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/server-3.svg width="100%" height="100%" %}</div>
# Model Training

1. Decide whether to train from scratch or fine-tune existing model
2. Choose loss function
3. Establish a simple baseline
4. Experiment with simple models
5. Switch to more complex models
6. Use an ensemble of models
7. Employ distributed training

---

## Key Concepts

- **Bias and Variance:** Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias can cause an algorithm to miss relevant relations between features and target outputs (underfitting). Variance refers to the amount by which a model would change if estimated using a different training dataset. High variance can cause an algorithm to model random noise in the training data, not the intended outputs (overfitting).

<div class="grid grid-cols-3 gap-2 items-center text-center">
  <div></div>
  <div>Low Bias</div>
  <div>High Bias</div>
  <div>Low Variance</div>
  <div>{% svg /assets/images/notes/machine-learning/bias-low-variance-low.svg  %}</div>
  <div>{% svg /assets/images/notes/machine-learning/bias-high-variance-low.svg  %}</div>
  <div>High Variance</div>
  <div>{% svg /assets/images/notes/machine-learning/bias-low-variance-high.svg  %}</div>
  <div>{% svg /assets/images/notes/machine-learning/bias-high-variance-high.svg  %}</div>
</div>

- **Bias-Variance Trade-off:** As you increase the complexity of your model, you will typically decrease bias but increase variance. On the other hand, if you decrease the complexity of your model, you increase bias and decrease variance.

- **Curse of Dimensionality:** As the number of features in a dataset increases, the volume of the feature space increases so fast that the available data becomes sparse. This makes it hard to have enough data to give meaningful results, leading to overfitting.

- **Learning Curve:** Model performance as a function of number of training examples, can be good for estimating if performance can be improved with more data

- **Overfitting and Underfitting:** overfitting occurs when a model learns the training data too well and can't generalise to unseen data, while underfitting happens when a model isn't powerful enough to model the training data.

<div class="grid grid-cols-3 gap-2 items-center text-center">
  <div></div>
  <div>Classification</div>
  <div>Regression</div>
  <div>Underfit</div>
  <div>{% svg /assets/images/notes/machine-learning/classification-underfit.svg  %}</div>
  <div>{% svg /assets/images/notes/machine-learning/regression-underfit.svg  %}</div>
  <div>Good Fit</div>
  <div>{% svg /assets/images/notes/machine-learning/classification-good-fit.svg  %}</div>
  <div>{% svg /assets/images/notes/machine-learning/regression-good-fit.svg  %}</div>
  <div>Overfit</div>
  <div>{% svg /assets/images/notes/machine-learning/classification-overfit.svg  %}</div>
  <div>{% svg /assets/images/notes/machine-learning/regression-overfit.svg  %}</div>
</div>

- **Universal Approximation Theorem:** A neural network with a single hidden layer can approximate any continuous function for inputs within a specific range

- **Vanishing/Exploding Gradients:** When training a deep neural network, if the gradient values become very small, they get "squashed" due to the activation functions resulting in vanishing gradients. When these small values get multiplied during backpropagation they can become near zero, which results in a lack of updates to the network weights and the training stalling. On the other hand, if the gradients become too large, they "explode", causing model weights to update too drastically and making model training unstable.

---

## Cross-validation[^sklearn] (CV)

- **K-fold:** divide samples into $$k$$ folds; train model on $$k-1$$ folds and evaluate using the left out fold
- **Leave One Out (LOO):** train model on all samples except one and evaluate using the left out sample
- **Stratified K-fold:** similar to K-fold, but each fold contains the same class balance as the full dataset
- **Group K-fold:** similar to K-fold, but ensure that groups (samples from the same data source) do not span different folds
- **Time Series Split:** to ensure only past observations are used to predict future observations, train model on first $$n$$ folds and evaluate on the $$n+1$$-th fold

---

## Hyperparameter Optimisation (HPO)

- **Grid search:** exhaustively search within bounds
- **Random search:** randomly search within bounds
- **Bayesian search:** modeled as Gaussian process

---

## Model Selection

1. Avoid the state-of-the-art trap; a state-of-the-art model only means that it performs better than existing models on some static datasets
2. Start with the simplest models, since they are: (i) easier to deploy; (ii) can be iterated on which aids interpretability; and (iii) can serve as a baseline
3. Avoid human biases in selecting models
4. Evaluate good performance now versus good performance later
5. Evaluate trade-offs, e.g. false positives vs. false negatives or compute requirement vs. accuracy
6. Understand your model's assumptions, e.g. prediction assumption, IID, smoothness, tractability, boundaries, conditional independence, normally distributed

---

## Neural Architecture Search[^cs329s] (NAS)

- **Search Space:** set of operations, (e.g. convolutions, fully-connected layers, pooling, etc.) and how they can be connected
- **Search Strategy:** random, reinforcement learning, evolution

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/trends-hot-flame.svg %}</div>
# Model Training: PyTorch[^pytorch]

## Activations

<div class="grid grid-cols-4 gap-2 items-center text-center">
  <div>Sigmoid<br />1 / (1 + e<sup>-x</sup>)</div>
  <div>{% svg /assets/images/notes/machine-learning/activation-sigmoid.svg  %}</div>
  <div>ReLU<br />max(0,x)</div>
  <div>{% svg /assets/images/notes/machine-learning/activation-relu.svg  %}</div>
  <div>Tanh<br />tanh(x)</div>
  <div>{% svg /assets/images/notes/machine-learning/activation-tanh.svg  %}</div>
  <div>Leaky ReLU<br />max(0.1x,x)</div>
  <div>{% svg /assets/images/notes/machine-learning/activation-leaky-relu.svg  %}</div>
</div>

---

## Debugging

- Overfit model on a subset of data
- Look out for exploding gradients (use gradient clipping)
- Turn on `detect_anomaly` so that any backward computation that generates `NaN` will raise an error.

---

## Distances

### Cosine Distance

$$d_{\text{Cosine}}(p, q) = 1 - \frac{p \cdot q}{\|p\| \|q\|}$$

Use for high-dimensional, sparse data where only the direction matters, e.g., document similarity, word embeddings.

### Manhattan Distance (L1 Norm or Taxicab Distance)

$$d_{\text{Manhattan}}(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

Use when movement is constrained to a grid, e.g. pathfinding, recommendation systems.

### Euclidean Distance (L2 Norm or Pairwise Distance)

$$d_{\text{Euclidean}}(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

Use when physical/geometric distance matters, e.g. dense data, clustering.

---

## Distributed Training

- **Data parallelism:** split the data across devices so that each device sees a fraction of the batch
- **Model parallelism:** split the model across devices so that each device runs a fragment of the model

---

## Initialisations

- **Kaiming:** sets the initial weights to account for the ReLU activation function by scaling the variance based on the number of input units, preventing vanishing gradients in deep networks.
- **Xavier:** initialises weights to keep the variance of activations uniform across layers by scaling the weights based on the number of input and output units, making it suitable for sigmoid and tanh activations.

---

## Layers

### Convolution Layers

- **Convolution:** Convolutional layers in PyTorch apply convolution operations to input data, using learnable filters (kernels) that slide over the input, detecting local features such as edges or textures in the case of images.

### Linear Layers

- **Linear:** A fully connected (dense) layer in PyTorch performs a linear transformation on the input by applying a weight matrix and adding a bias vector:

  $$\text{output} = \text{input} \times W^T + b$$

  It's typically used in the final layers of a neural network, allowing every input feature to connect to every output feature.

### Pooling Layers

- **Average pool:** takes the average of values within a pooling window, preserving spatial information by smoothing feature maps.
- **Max pool:** selects the maximum value within a pooling window, retaining the most prominent features while reducing dimensionality.
- **Adaptive max pool:** adjusts the pooling window size dynamically to output a fixed-sized feature map, regardless of input size.
- **Fractional max pool:** pools using non-integer strides, allowing for more flexible downsampling that preserves more information in deeper layers.

### Recurrent Layers

- **Recurrent Neural Network (RNN):** The basic RNN layer processes sequence data by maintaining a hidden state across time steps. It’s useful for tasks involving time-series, language, or any sequential data, however, they suffer from the vanishing gradient problem, which can hinder learning over long sequences.
- **Long short-term memory (LSTM):** LSTMs improve upon standard RNNs by using three gates (input gate, forget gate, and output gate) and a memory cell to selectively retain or discard information over time, which helps avoid the vanishing gradient problem.
- **Gated recurrent unit (GRU):** GRUs improve upon standard RNNs by using two gates (reset gate and update gate) to control the flow of information, which helps solve the vanishing gradient problem, making it easier to capture dependencies over longer sequences without needing separate memory cells like LSTMs.

### Transformer Layers

- **Transformer:** The transformer layer is designed to handle sequential data without relying on recurrent structures. It uses a self-attention mechanism to learn relationships between different positions in the sequence, making it highly parallelisable and better suited for long-range dependencies than RNN-based models.
- **Transformer Encoder:** The encoder is one half of the transformer architecture, which processes input sequences into a rich set of representations. It consists of multiple layers of multi-head self-attention and position-wise feed-forward networks, which allow the model to understand context across the entire sequence.
- **Transformer Decode:** The decoder is the other half of the transformer architecture, used for tasks like sequence-to-sequence modeling (e.g., machine translation). It generates output sequences by attending to both the encoder's output and previous tokens of the output sequence, enabling it to produce context-aware predictions.

---

## Loss Functions

- **Cross-Entropy** measures how close the predicted probability distribution is with the true distribution. It is widely used in classification tasks, especially for multi-class problems, where it penalizes incorrect classifications based on how confident the model was in its predictions.

  $$l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}$$

- **Connectionist Temporal Classification (CTC)** is used where the alignment between input and output sequences is unknown, such as in speech recognition or handwriting recognition, where the timing of outputs may not correspond directly to the inputs.

- **Huber Loss** is a combination of Mean Squared Error and Mean Absolute Error that is less sensitive to outliers than MSE. It behaves as MSE when the error is small and as MAE when the error is large.

  $$l_n = \begin{cases} \frac{1}{2}(x_n - y_n)^2 & \text{if } |x_n - y_n| < \delta, \\ \delta(|x_n - y_n| - \frac{1}{2}\delta) & \text{otherwise,}\end{cases}$$

  where $$\delta$$ is a threshold defining when to switch between the two behaviors.

- **Kullback–Leibler (KL) Divergence** measures how one probability distribution diverges or is different from a second, expected probability distribution. It is used for comparing probability distributions, often in generative models.

  $$L(y_{\text{pred}},\ y_{\text{true}})  = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})$$

- **L<sub>1</sub> Loss** measures the sum of the absolute values of the difference between the predicted and actual values.

  $$l_n = \sum \left| x_n - y_n \right|$$

- **Mean Absolute Error** measures the average $$L_1$$ loss across a set of $$N$$ examples. It is robust to outliers and focuses on minimizing large deviations.

  $$l_n = \frac{1}{N} \sum \left| x_n - y_n \right|$$

- **L<sub>2</sub> Loss** measures the sum of the squared difference between the predicted and actual values.

  $$l_n = \sum \left( x_n - y_n \right)^2$$

- **Mean Squared Error** measures the average of L_2 losses across a set of $$N$$ examples. It penalizes larger errors more than smaller ones.

  $$l_n = \frac{1}{N} \sum \left( x_n - y_n \right)^2$$

- **Root Mean Squared Error** measures the square root of the mean squared error (MSE). It penalizes larger errors more than smaller ones.

  $$l_n = \sqrt{ \frac{1}{N} \sum \left( x_n - y_n \right)^2 }$$

- **Negative Log Likelihood (NLL)** measures the disagreement between the true labels and the predicted probability distributions, assigning a high penalty to incorrect classifications where the predicted probability was high.

  $$l_n = - w_{y_n} x_{n,y_n}$$

---

## Normalisation

<div class="grid grid-cols-4 gap-2 items-center text-center">
  <div>Batch Norm</div>
  <div>{% svg /assets/images/notes/machine-learning/norm-batch.svg  %}</div>
  <div>Layer Norm</div>
  <div>{% svg /assets/images/notes/machine-learning/norm-layer.svg  %}</div>
  <div>Group Norm</div>
  <div>{% svg /assets/images/notes/machine-learning/norm-group.svg  %}</div>
  <div>Instance Norm</div>
  <div>{% svg /assets/images/notes/machine-learning/norm-instance.svg  %}</div>
</div>

---

## Optimisers

- **Adagrad:** adapts the learning rate for each parameter based on its past gradients, making frequent updates smaller and rare updates larger.
- **Adam:** combines the benefits of momentum and RMSProp, using moving averages of both gradients and squared gradients to adapt learning rates.
- **Momentum:** accelerates gradients in the relevant direction by combining the current gradient with a fraction of the previous gradient, helping to avoid local minima.
- **RMSProp:** adjusts the learning rate for each parameter based on the moving average of squared gradients, preventing large oscillations in the update step.
- **Stochastic Gradient Descent (SGD):** updates parameters using only a random subset of data, reducing computation per update but introducing noise to the gradient estimation.

---

## Performance Tuning[^pytorch][^pytorch_lightning]

- Enable asynchronous data loading and augmentation using `num_workers > 0` and `pin_memory = True`
- Disable bias for convolutions before batch norms
- Use learning rate scheduler
- Use mixed precision
- Accumulate gradients by running a few small batches before doing a backward pass
- Saturate GPU by maxing-out batch size (downside: higher batch sizes may cause training to get stuck in local minima)
- Use Distributed Data Parallel (DDP) for multi-GPU training
- Clip gradients to avoid exploding gradients
- Disable gradient calculation for val/test/predict

---

## Regularisation

### Augmentation

- **Image:** random crop, saturation, flip, rotation, translation, perturb using random noise
- **Text:** swap with synonyms, add degree adverbs, perturb with random word replacements

### Data Synthesis

- **Image:** mixup (inputs and labels are linear combination of multiple classes)
- **Text:** template-based, language model-based

### Dropout

Randomly zeroes some elements of the input tensor with probability $$p$$, forcing the model to learn redundant representations and reducing the risk of overfitting.

### Early Stopping

Stops training when the model's performance on a validation set stops improving, preventing overfitting by stopping before the model starts memorizing noise.

### L1 Regularisation (Lasso)

Adds a penalty equal to the absolute value of the magnitude of the coefficients to the loss function. This results in sparse solutions by driving less important feature weights to zero, making it useful for feature selection

### L2 Regularisation (Ridge)

Adds a penalty equal to the square of the magnitude of the coefficients to the loss function. This discourages large weights and helps in reducing overfitting without driving coefficients to exactly zero.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/diagram-down.svg width="100%" height="100%" %}</div>
# Loss Curves

Reading the shape of the loss curve is often the fastest way to diagnose a training run.

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-ideal-loss.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Ideal loss** shows healthy convergence. Smooth decline to a stable minimum.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-oscillating-loss.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Oscillating loss** due to unstable updates. Lower the learning rate; clean the data.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-sharp-rise-in-loss.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Sharp rise in loss** due to bad batch. Check for NaNs/Inf, invalid labels, and outliers.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-overfitting.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Overfitting** widens the train/val gap. Add regularization or early stopping; simplify.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-chaotic-loss.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Chaotic loss** due to poor shuffling. Inspect the sampler and data pipeline.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-high-loss-plateau.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**High loss plateau** due to poor fit. Confirm it can overfit a tiny subset first.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-little-or-no-learning.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Little or no learning** due to broken gradient flow. Check gradients, updates, and label wiring.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-underfitting.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Underfitting** leaves loss high. Train longer or add features/capacity.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-noisy-validation-loss.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Noisy validation loss** due to eval variance. Use a larger, fixed validation set.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-step-down-after-lr-decay.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Step down after LR decay** is expected. Should match an LR decay or unfreeze.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-periodic-loss-spikes.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Periodic loss spikes** due to a recurring pattern. Align with LR cycles or periodic data.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-validation-loss-cliff.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Validation loss cliff** due to a pipeline mismatch. Check preprocessing and leakage.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-nan-or-inf-loss.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**NaN or Inf loss** due to numerical failure. Check finiteness, overflow, and precision.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-loss-regression.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Loss regression** due to a regime change. Check LR spikes, resets, or forgetting.
</div>
</div>

<div class="grid grid-cols-[1fr_3fr] items-center gap-4 mb-4">
<div>{% svg /assets/images/notes/machine-learning/loss-curve-suspiciously-fast-convergence.svg class="w-full aspect-square" %}</div>
<div markdown="1">
**Suspiciously fast convergence** due to likely leakage. Check features and held-out data.
</div>
</div>

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/search.svg width="100%" height="100%" %}</div>
# Model Evaluation

Evaluating machine learning model performance.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/earth-2.svg width="100%" height="100%" %}</div>
# Model Evaluation: Responsible AI

## Compactness

Reduces memory footprint and increases computation speed

- **Quantisation:** reduce model size by using fewer bits to represent parameters
- **Knowledge distillation:** train a small model (student) to mimic the results of a larger model (teacher)
- **Pruning**: remove nodes or set least useful parameters to zero
- **Low-ranked factorisation:** replace convolution filters with compact blocks

---

## Explainability

- **Integrated Gradients:** compute the contribution of each feature to a prediction by integrating gradients over the path from the baseline
- **LIME (Local Interpretable Model-agnostic Explanations):** creates a simpler, interpretable model around a single prediction to explain how the model behaves at that specific instance.
- **Sampled Shapley:** estimates the contribution of each feature by averaging over subsets of features sampled from the input data.
- **SHAP (SHapley Additive exPlanations):** assigns each feature an importance value for a particular prediction, based on the concept of Shapley values from cooperative game theory
- **XRAI (eXplanation with Ranked Area Integrals):** segments an input image and ranks the segments based on their contribution to the model's prediction

---

## Fairness

- Slice-based evaluation, e.g. when working with website traffic, slice data among: gender, mobile vs. desktop, browser, location
- Check for consistency over time
- Determine slices by heuristics or error analysis

---

## Robustness

- **Determinism Test:** ensure same outputs when predicting using same model
- **Retraining Invariance Test:** ensure similar outputs when predicting using re-trained model
- **Perturbation Test:** ensure small changes to numeric inputs don't cause big changes to outputs
- **Input Invariance Test:** ensure changes to certain inputs don't cause changes to outputs
- **Directional Expectation Test:** ensure changes to certain inputs cause predictable changes to outputs
- **Ablation Test:** ensure all parts of the model are relevant for model performance
- **Fairness Test:** ensure different slices have similar model performance
- **Model Calibration Test:** ensure events should happen according to the proportion predicted

---

## Safety

- **Alignment:** ensuring that AI systems’ goals and behaviors are in accordance with human values and intentions, preventing them from acting in ways that could harm or be misaligned with user interests.
- **Existential Risk:** the potential risk that AI systems could lead to catastrophic outcomes that threaten the long-term survival or flourishing of humanity, such as uncontrolled superintelligent AI.
- **Red Teaming:** experts simulate potential attacks on a system to identify vulnerabilities, test defenses, and improve system security before actual attackers do.
- **Reward Hacking:** when an AI system finds unintended ways to maximize its reward function, leading to harmful or suboptimal outcomes that violate the intended goals.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/analytics-pie-2.svg width="100%" height="100%" %}</div>
# Model Evaluation: Metrics

## Offline Metrics (Before Deployment)

### Baselines

- Predict at random (uniformly or following label distribution)
- Zero rule baseline (always predict the most common class)
- Simple heuristics
- Human baseline
- Existing solutions

### Classification

- **Confusion Matrix**

  |             | Class 1             | Class 2             |
  | ----------- | ------------------- | ------------------- |
  | **Class 1** | True-positive (TP)  | False-positive (FP) |
  | **Class 2** | False-negative (FN) | True-negative (TN)  |

  {: .table .table-striped }

- **Type I error:** FP
- **Type II error:** FN
- **Precision:** TP / (TP + FP), i.e. a classifier's ability not to label a negative sample as positive
- **True-positive rate (Recall or Sensitivity):** TP / (TP + FN), i.e. a classifier's ability to find all positive samples
- **True-negative rate (Specificity):** TN / (TN + FP), i.e. a classifier's ability to identify all negative samples
- **False-positive rate:** FP / (FP + TN), i.e. a classifier's inability to find all negative samples
- **F1 score:** 2 × precision × recall / (precision + recall), i.e. the harmonic mean of precision and recall
- **Precision-recall curve:** trade-off between precision and recall, a higher PR-AUC indicates a more accurate model
- **Receiver operator characteristic (ROC) curve:** trade-off between true-positive rate (recall) and false-positive rate, a higher ROC-AUC indicates a model better at distinguishing positive and negative classes

### Regression

- **Mean squared error (MSE):** average of the squared differences between the predicted and actual values, emphasising larger errors
- **Mean absolute error (MAE):** average of the absolute differences between the predicted and actual values, treating all errors equally
- **Root mean square error (RMSE):** square root of the MSE, providing error in the same units as the predicted and actual values and emphasizing larger errors like MSE

### Object Recognition

- **Intersection over union (IOU):** ratio of overlap area with union area

### Ranking

- **Recall@k:** proportion of relevant items that are included in the top-k recommendations
- **Precision@k:** proportion of top-k recommendations that are relevant
- **Mean reciprocal rank (MRR):** $$\frac{1}{m} \sum_{i=1}^m \frac{1}{\textrm{rank}_i}$$, i.e. where is the first relevant item in the list of recommendations?
- **Hit rate:** how often does the list of recommendations include something that's actually relevant?
- **Mean average precision (mAP):** mean of the average precision scores for each query
- **Diversity:** measure of how different the recommended items are from each other
- **Coverage:** what's the percentage of items seen in training data that are also seen in recommendations?
- **Cumulative gain (CG):** $$\sum_{i=1}^p rel_i$$, i.e. sum of relevance scores obtained by a set of recommendations
- **Discounted cumulative gain (DCG):** $$\sum_{i=1}^p \frac{\textrm{rel}_i}{\log_2(i+1)}$$, i.e. CG discounted by position
- **Normalised discounted cumulative gain (nDCG):** $$\frac{\textrm{DCG}_p}{\textrm{IDCG}_p}$$, i.e. extension of CG that accounts for the position of the recommendations (discounting the value of items appearing lower in the list), normalised by maximum possible score

### Image Generation

- **Fréchet Inception Distance (FID):** measures the distance between the feature distributions of generated images and real images using a pre-trained Inception model. Used to assess the quality and realism of generated images by comparing them with real-world images.
- **Inception score:** calculates how confidently the model can classify generated images and measures how diverse the generated images are across categories. Used to assess how well the generated images align with recognizable classes, balancing image quality and diversity.

### Fluency

- **Perplexity:** measures how well a language model predicts the next token in a sequence, representing the model's uncertainty. Used to evaluate the fluency and coherence of a language model in predicting sequences of words or tokens.

### Summarisation[^yan24]

- **Factual consistency:** finetune a Natural Language Inference (NLI) model to predict whether the hypothesis sentence is entailed by (logically flows from), neutral to or contradicts the premise sentence.

- **Sentiment consistency:** for each key aspect, does the summary accurately reflect the sentiment for each key aspect?

- **Aspect relevance:** does the summary cover the main topics discussed?

- **Length adherence:** does the summary meet a word or character limit?

### Translation[^yan24]

- **Character n-gram F-score (chrF):** compute the precision and recall of character n-grams between the machine translation (MT) and the reference translation.

- **BLEURT:** compute the similarity between the machine translation (MT) and the reference translation using a pre-trained BERT model.

### Copyright Regurgitation[^yan24]

- **Exact regurgitation:** compute the length of the longest common subsequence (LCS) between the generated text and the copyright reference, normalised by the length of the input prompt.

- **Near-exact reproduction:** copmute the edit distance between the generated text and the copyright reference, normalised by the length of the input prompt.

### Toxicity[^yan24]

- **Toxicity score:** proportion of generated output that is classified as harmful, offensive or inappropriate.

---

## Online Metrics (After Deployment)[^mlsystemdesign]

### Examples

- **Event recommendation:** conversion rate, bookmark rate, revenue lift
- **Friend recommendation:** number of requests per day, number of requests accepted per day
- **Harmful content detection:** prevalence, harmful impressions, valid appeals, proactive rate, user reports per harmful class
- **Video recommendations:** click-through-rate, video completion rate, total watch time, explicit user feedback

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/space-rocket-flying.svg width="100%" height="100%" %}</div>
# Model Deployment

## Continual Learning

- Continually adapt models to changing data distributions
- Faced with challenges of access to fresh data, continuous evaluation and algorithms suited to fine-tuning and incremental learning

### Stages

1. **Manual, stateless retraining:** initial manual workflow.
2. **Automated retraining:** requires writing a script to automate workflow and configure infrastructure automatically, good availability and accessibility of data, and a model store to automatically version and store all the artefacts needed to reproduce a model.
3. **Automated, stateful retraining:** requires reconfiguring the updating script and the ability to track data and model lineage.
4. **Continual learning:** requires mechanism to trigger model updates and a pipeline to continually evaluated model updates.

---

## Deployment Strategies (B to replace A)

- **Recreate strategy:** stop A, start B
- **Ramped strategy:** shift traffic from A to B behind same endpoint
- **Blue/green:** shift traffic from A to B using different endpoint

---

## ML-specific Failures[^cs329s]

Train-serving skew is when a model performs well during development but poorly after production. It can be caused by:

### Upstream Drift

Caused by a discrepancy between how data is handled in the training and serving pipelines (should log features at serving time)[^ml_rules]

### Data Distribution Shifts

Model may perform well when first deployed, but poorly over time (can be sudden, cyclic or gradual).

- **Feature/covariate shift:** change in the distribution of input data, $$P(X)$$, but relationship between input and output, $$P(Y \vert X)$$, remains the same.
  - In training, can be caused by changes to data collection, e.g. if early data is from urban customers, and later data comes from rural customers.
  - In production, can be caused by changes to external factors, e.g. when predicting sales from weather, if weather patterns change (more rainy days), but the relationship between weather and sales remains constant (rainy days always lead to fewer sales).
- **Label shift:** change in the distribution of output labels, $$P(Y)$$, but relationship between output and input, $$P(X \vert Y)$$, remains the same.
  - E.g. when predicting diseases, if a disease becomes more common, but symptoms for each disease remains constant.
- **Concept drift:** change in the relationship between input and output, $$P(Y \vert X)$$, but the distribution of input data, $$P(X)$$, remains the same.
  - E.g. when predicting rain from cloud patterns, if the cloud patterns remain the same but their association with rain changes (maybe due to climate change).

### Degenerate Feedback Loops

When predictions influence the feedback, which is then used to extract labels to train the next iteration of the model,

### Examples

- **Recommender system:** originally, A is ranked marginally higher than B, so the model recommends A. After a while, A is ranked much higher than B. Can be detected using Average Recommended Popularity (ARP) and Average Percentage of Long Tail Items (APLT).
- **Resume screening:** originally, model thinks X is a good feature, so the model recommends resume with X. After a while, hiring managers only hires people with X and model confirms X is good. Can be mitigated using randomisation and positional features.

---

## Model Monitoring

Model monitoring is essential because while traditional software systems fail explicitly (error messages), Machine Learning systems fail silently (bad outputs)

### Operation-related Metrics

- Latency
- Throughput
- Requests per minute/hour/day
- Percentage of successful requests
- CPU/GPU utilisation
- Memory utilisation
- Availability

### ML-related Metrics[^cs329s]

- Feature and label statistics, e.g. mean, median, variance, quantiles, skewness, kurtosis, etc.
- Task-specific online metrics

---

## Myths[^huyen22]

1. You only deploy one or two models at a time
2. If we don't do anything, model performance remains the same
3. You won't need to update your models as much
4. Most ML engineers don't need to worry about scale

---

## Testing Strategies

- **Canary:** targeting small set of users with latest version.
- **Shadow:** mirror incoming requests and route to shadow application.
- **A/B:** route to new application depending on rules or contextual data.
- **Interleave:** mix recommendations from A and B and see which recommendations are clicked on.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/neural-network.svg width="100%" height="100%" %}</div>
# Models

Overview of machine learning models.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/brain-1.svg width="100%" height="100%" %}</div>
# Models: Supervised Learning[^isl][^sklearn]

Supervised learning models make predictions after seeing lots of data with the correct answers. The model discovers the relationship between the data and the correct answers.

## Regression

Regression models predict a numeric value.

### Linear Regression

$$\hat{y}(x) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon$$

- Linear regression models the relationship between the target and predictors as a straight line.
- Parameters are estimated by minimising the Residual Sum of Squares (RSS): $$RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- Use when the relationship between features and the target is approximately linear.

### Regression Trees

$$\hat{y}(x) = \frac{1}{|R_j|} \sum_{i \in R_j} y_i \quad \text{if} \, x \in R_j$$

- Regression trees split the feature space into regions and predict the average of observations within each region.
- The tree splits are chosen to minimise the RSS in the resulting regions.
- Use when the relationship between features and the target is non-linear, and you prefer a model that is easy to interpret.

### Support Vector Regressor (SVR)

$$\hat{y}(x) = w^T x + b \quad \text{for points within the} \, \epsilon \, \text{margin}$$

- The Support Vector Regressor (SVR) fits a margin around the data points and penalises observations that fall outside the margin.
- Parameters are estimated by solving a quadratic optimisation problem to maximize the margin and penalize points outside the margin.
- Use when the relationship between features and the target is complex and when outliers need to be managed by a margin.

---

## Classification

Classification models predict the likelihood that something belongs to a category.

### Nearest Neighbours

$$\hat{y}(x) = \arg\max_{c \in C} \sum_{i \in N_k(x)} I(y_i = c)$$

- The k-Nearest Neighbours (k-NN) algorithm assigns the class that is most frequent among the $$k$$ closest observations in the feature space.
- No explicit training; classification is based on the majority class among the nearest neighbors using distance metrics.
- Use when there's no assumption about the underlying distribution of the data and when simplicity is desired.
- Algorithm can either be exact or approximate:
  - Clustering-based (e.g. HNSW): group points into clusters based on similarity, then search for neighbors only within relevant clusters, reducing the search space.
  - Locality-sensitivity hashing (LSH): uses a hash function to map similar points to the same bucket, enabling efficient approximate search in high-dimensional spaces.
  - Tree-based (e.g. $$k$$-d Tree): organize data into a hierarchical structure, allowing efficient nearest-neighbor search by pruning partitions that cannot contain the nearest neighbors.

### Logistic Regression

$$\hat{y}(x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p)}}$$

- Logistic regression models the probability that an observation belongs to a particular class, with the logistic function constraining the output between 0 and 1.
- Parameters are estimated by Maximum Likelihood Estimation (MLE), which maximises the probability of the observed data.
- Use when the relationship between the features and the probability of class membership is approximately linear on the log-odds scale.

### Linear Discriminant Analysis (LDA)

$$\hat{y}(x) = \arg\max_k \left( \log(\pi_k) - \frac{1}{2} \log |\Sigma| - \frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k) \right)$$

- Linear Discriminant Analysis (LDA) assumes that different classes generate data based on multivariate normal distributions with class-specific means and a shared covariance matrix.
- Parameters are estimated by calculating class means, the shared covariance matrix, and the prior probabilities for each class.
- Use when the data is approximately normally distributed, and the classes have similar covariance matrices.

### Classification Trees

$$\hat{y}(x) = \arg\max_{c \in C} \hat{p}_{mc} \quad \text{where} \, x \in R_m$$

- Classification trees split the feature space into regions where each region is assigned the most common class.
- The tree splits are chosen to maximise information gain, using metrics such as Gini impurity or entropy.
- Use when you need a simple, interpretable model that can handle both non-linear relationships and categorical features.

### Support Vector Classifier (SVC)

$$\hat{y}(x) = \text{sign}(w^T x + b)$$

- The Support Vector Classifier (SVC) finds the hyperplane that best separates the data into classes by maximising the margin between them.
- Parameters are estimated by solving a quadratic optimization problem to maximize the margin while allowing for some misclassification via slack variables.
- Use when the data is complex and not linearly separable, and you need a robust classifier with regularization to avoid overfitting.

---

## Variants

### Active Learning

Model selects the most informative samples to be labeled by an oracle (e.g. a human expert) to improve learning efficiency.

### Contrastive Learning

Learning by distinguishing between similar and dissimilar data points.

### Few-Shot Learning

Learning from very few labeled examples, often using meta-learning.

### Meta-Learning

Learning across multiple tasks to generalize better to new, unseen tasks with limited data.

### Self-Supervised Learning

Model generates its own labels from unlabeled data using pretext tasks.

### Semi-supervised Learning

Learning from a dataset with a small amount of labeled data and a large amount of unlabeled data, leveraging structure in the unlabeled examples.

### Weakly Supervised Learning

Learning from data with imperfect, noisy, or incomplete labels.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-9.svg width="100%" height="100%" %}</div>
# Models: Unsupervised Learning

Unsupervised learning involves finding patterns and structure in input data without any corresponding output data.

## Clustering

Clustering is used to group observations into clusters, where observations within the same cluster are more similar to each other than to those in different clusters.

### Hierarchical Clustering

$$ \hat{y}(x) = \text{Cluster assignment based on dendrogram} $$

- Hierarchical clustering builds a tree-like structure (dendrogram) of nested clusters either from individual points up (agglomerative) or from one large cluster down (divisive).
- No explicit parameter estimation required; clusters are formed by successively merging or splitting based on a linkage criterion (e.g., complete, single, or average linkage).
- Use when you want to explore data structure at different levels of granularity and do not need to pre-specify the number of clusters.

### K-Means Clustering

$$ \hat{y}(x) = \arg\min\_{k} ||x - \mu_k||^2 $$

- K-Means Clustering partitions the data into $$K$$ clusters by assigning each point to the nearest cluster centroid and updating centroids iteratively to minimise the within-cluster sum of squares (WCSS).
- Centroids are updated iteratively by minimising the within-cluster sum of squares (WCSS): $$WCSS = \sum_{k=1}^{K} \sum_{i \in C_k} \lVert x_i - \mu_k \rVert ^2$$, where $$\mu_k$$ is the centroid of cluster $$C_k$$.
- Use when you know the number of clusters in advance and the clusters are roughly spherical and evenly sized.

### Latent Dirichlet Allocation (LDA)

$$ \hat{y}(x) = \arg\max\_{z_i} P(z_i | d, \theta_d, \phi_z) $$

- Latent Dirichlet Allocation (LDA) assumes that documents are mixtures of topics, and topics are distributions over words. It assigns each word in a document to a latent topic.
- Parameters (topic distributions and word distributions) are estimated using variational inference or Gibbs sampling. The key parameters are $$\theta_d$$ (the distribution of topics in document $$d$$ and $$\phi_z$$ (the distribution of words in topic $$z$$).
- Use LDA when you want to uncover latent topics in a large corpus of text and when documents are assumed to have multiple topics.

---

## Dimensionality Reduction

Reduce the number of features (or dimensions) in the data while retaining as much of the variance or structure as possible.

### Principal Components Analysis (PCA)

$$ \hat{y}(x) = Z*1 = \phi*{11} X*1 + \phi*{12} X*2 + \ldots + \phi*{1p} X_p $$

- Principal Components Analysis (PCA) finds a new set of uncorrelated variables (principal components) that successively explain the maximum variance in the data.
- Principal components are the eigenvectors of the covariance matrix $$\Sigma$$, and the corresponding eigenvalues represent the variance explained by each component. PCA maximizes the variance explained: $$\text{Maximize } \text{Var}(Z_k) = \phi_k^T \Sigma \phi_k$$, subject to $$\phi_k^T \phi_k = 1$$, where $$Z_k$$ is the $$k$$-th principal component.
- Use when you need to reduce dimensionality while preserving as much variance as possible, particularly when features are correlated.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/chess-figures.svg width="100%" height="100%" %}</div>
# Models: Reinforcement Learning[^spinning_up]

Reinforcement Learning models help agents learn the best action to take in an environment in order to achieve its goal.

---

## Key Concepts

### Agent-Environment Interaction

The <u>environment</u> is the world where the <u>agent</u> operates. At each step, the agent perceives a <u>state</u> (complete world description) or an <u>observation</u> (partial information) and selects an <u>action</u> from the <u>action space</u> (discrete or continuous). A <u>policy</u> is a rule the agent follows to choose actions, aiming to maximise <u>rewards</u>.

<figure class="flex flex-col items-center">
```mermaid
graph TD
    Agent -- Action (Aₜ) --> Environment
    Environment -- Reward (Rₜ) --> Agent
    Environment -- State (Sₜ) --> Agent
```
</figure>

### Terminology

- **Trajectory**: A sequence of states and actions, $$\tau = (s_0, a_0, s_1, a_1, \ldots)$$.
- **Reward Function**: Determines the reward based on the state and action, $$r_t = R(s_t, a_t)$$.
- **Return**: Cumulative reward over time. It can be finite-horizon (sum over a fixed window) or infinite-horizon discounted return ($$\gamma$$-discounted future rewards).
- **Value Function**: Measures the expected return starting from a state or state-action pair and acting according to a particular policy forever. The Bellman Equation captures the recursive nature of value estimation:
  $$V^\pi(s) = R^\pi(s) + \gamma \sum_{s' \in S} P^\pi(s' \vert s) V^\pi(s')$$
- **Policy**: A policy $$\pi$$ is a rule or function that the agent uses to decide which action to take given a state. It can be deterministic, mapping a state to a specific action, $$\mu(s)$$, or stochastic, mapping a state to a probability distribution over actions, $$\pi(a \vert s)$$.
- **Optimisation Goal**: The agent learns to select a policy that maximises expected return over time.

---

## Kinds of Reinforcement Learning Algorithms

### Model-Free

Does not use a model of the environment, which is often unavailable anyway. It focuses on learning through direct interaction.

- **Policy Optimisation**: Directly optimises the policy typically by estimating the expected future rewards. E.g. A3C, PPO.
- **Q-Learning**: Learns an action-value function to estimate the optimal value for each action. E.g. C51, DQN.

### Model-Based

Uses a model to predict state transitions and rewards, enabling planning ahead. It gains sample efficiency but is prone to bias if the model is inaccurate.

- **Learn the Model**: The agent learns the environment's dynamics from experience. This enables planning but can introduce bias due to model inaccuracies. E.g. I2A, MBMF, MBVE.
- **Given the Model**: The agent is provided with an accurate environment model, allowing for optimal planning. This is rare in real-world scenarios but useful in controlled environments like games. E.g. AlphaZero.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/rating-five-star.svg width="100%" height="100%" %}</div>
# Models: Recommender Systems[^mlsystemdesign]

## Behavioural principles

- Similar items are symmetric, e.g. white polo shirts
- Complementary items are asymmetric, e.g. buying a television, suggest a HDMI cable

---

## Rule-based

Rule-based recommender systems rely on predefined rules and heuristics to make recommendations based on explicit logic and user behavior patterns. These systems do not involve machine learning but instead use a fixed set of if-then conditions to guide recommendations.

---

## Embedding-based

### Content-based Filtering

Item feature similarities

- **Pros:** ability to recommend new videos, ability the capture unique user interests
- **Cons:** difficult to discover a user's new interests, requires domain knowledge to engineer features
- **Models:** image embeddings, text embeddings

### Collaborative Filtering

User-to-user similarities or item-to-item similarities

- **Pros:** no domain knowledge needed, easy to discover users' new areas of interest, efficient, training/serving speed
- **Cons:** cold-start problem, cannot handle niche interests
- **Models:** matrix factorization

### Hybrid Filtering

Parallel or sequential combination of content-based and collaborative filtering

- **Pros:** combines strengths of both methods for better recommendations
- **Cons:** more complex to implement, training/serving speed
- **Models:** two-tower neural network

---

## Learning-to-Rank

- **Point-wise:** model takes each item individually and learns to predict an absolute relevancy score
- **Pair-wise:** model takes two ranked items and learns to predict which item is more relevant (RankNet, LambdaRank, LambdaMART)
- **List-wise:** model takes optimal ordering of items and learns the ordering (SoftRank, ListNet, AdaRank)
</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Models: Ensembles

## Bagging (Bootstrap Aggregation)

Reduces model variance by training identical models in parallel on different data subsets (random forests).

- **Pros:** reduces overfitting, parallel training means little increase in training/inference time
- **Cons:** not helpful for underfit models

---

## Boosting

Reduces model bias and variance by training several weak classifiers sequentially (Adaboost, XGBoost).

- **Pros:** reduces bias and variance
- **Cons:** slower training and inference

---

## Stacking (Stacked Generalisation):

Reduces model bias and variance by training different models in parallel on the same dataset and using a meta-learner model to combine the results.

- **Pros:** reduces bias and variance, parallel training means little increase in training/inference time
- **Cons:** prone to overfitting

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/educative-toys-maths.svg width="100%" height="100%" %}</div>
# Models: Tasks

## Audio

- **Audio generation**: WaveNet, Tacotron, Jukebox
- **Classification**: VGGish, SoundNet, YAMNet
- **Speaker diarization**: X-vector, DIHARD, VBx
- **Speaker identification**: SincNet, x-vector, ECAPA-TDNN
- **Speech recognition**: DeepSpeech, Wav2Vec, RNN-T (Recurrent Neural Network Transducer)
- **Source separation**: Demucs, Conv-TasNet, Open-Unmix
- **Text-to-speech**: Tacotron 2, FastSpeech, WaveGlow

---

## Computer Vision

- **3D reconstruction**: AtlasNet, DeepVoxels, NeRF (Neural Radiance Fields)
- **Action recognition**: I3D (Inflated 3D ConvNet), C3D (Convolutional 3D), SlowFast Networks
- **Classification**: AlexNet, Inception, ResNet, VGG
- **Depth estimation**: Monodepth, DPT (Dense Prediction Transformers), SfM-Net (Structure from Motion Network)
- **Image captioning**: Show and Tell, Show, Attend and Tell, OSCAR
- **Image denoising**: DnCNN, N2V (Noise2Void), BM3D
- **Image inpainting**: DeepFill, Context Encoders, LaMa
- **Image-to-image translation**: Pix2Pix, CycleGAN, UNIT (Unified Image Translation)
- **Object detection**: Faster R-CNN, YOLO, SSD (Single Shot Multibox Detector)
- **Object tracking**: SORT, DeepSORT, SiamRPN (Siamese Region Proposal Network)
- **Optical character recognition (OCR)**: CRNN (Convolutional Recurrent Neural Network), Tesseract, Rosetta
- **Optical flow**: FlowNet, PWC-Net, RAFT (Recurrent All-Pairs Field Transforms)
- **Pose estimation**: OpenPose, PoseNet, HRNet
- **Semantic segmentation**: U-Net, DeepLab, SegNet
- **Super-resolution**: SRGAN, ESRGAN (Enhanced SRGAN), VDSR (Very Deep Super-Resolution)
- **Text-to-image generation**: DALL-E, Parti, Imagen
- **Visual odometry**: ORB-SLAM, VISO2, DeepVO

---

## Graphs

- **Community detection**: Louvain, Label Propagation, Infomap
- **Graph classification**: Graph Convolutional Networks (GCNs), GraphSAGE, GIN (Graph Isomorphism Network)
- **Link prediction**: Node2Vec, DeepWalk, SEAL
- **Node prediction**: Graph Attention Networks (GAT), GCN, GraphSAGE

---

### Natural Language Processing

- **Classification**: BERT, RoBERTa, XLNet
- **Question answering**: BERT, T5, ALBERT
- **Language modeling**: GPT, GPT-2, GPT-3, Transformer
- **Machine translation**: Transformer, MarianMT, mBART
- **Named entity recognition**: BERT, Flair, SpaCy's CNN model
- **Part-of-speech tagging**: BiLSTM-CRF, BERT, Flair
- **Sentiment analysis**: BERT, XLNet, RoBERTa
- **Text generation**: GPT-2, GPT-3, T5
- **Text summarisation**: BART, PEGASUS, T5

---

## Miscellaneous

- **Anomaly detection**: Isolation Forest, One-Class SVM, Autoencoders
- **Autonomous driving**: MobileNet, YOLO (You Only Look Once), PointPillars
- **Code generation**: GPT-3, Codex, AlphaCode
- **Time-series forecasting**: ARIMA, Prophet, LSTM (Long Short-Term Memory)

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/picture-landscape.svg width="100%" height="100%" %}</div>
# Models: Image Generation Models (TODO)

Deep learning architectures designed to synthesize realistic or stylized images from various inputs such as noise, text, or existing images.

- **Diffusion Models** generate images by iteratively denoising a random noise input.

- **Low-Rank Adaptation (LoRA)** is a fine-tuning technique that freezes the pre-trained model weights and injects trainable low-rank matrices into each transformer block.

- **DreamBooth** fine-tunes all the parameters in the diffusion model while keeping the text transformer frozen.

- **Variational Autoencoders (VAEs)** enable diffusion to operate in a compressed latent space instead of raw pixel space.

- **Text Inversion (Negative Embeddings)** allows models to learn new concepts from a small number of sample images by optimizing a new word embedding token for each concept.

- **Low-Rank Conditioning for Regularization in Image Synthesis (LyCORIS)** extends LoRA by incorporating additional conditioning mechanisms.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/paragraph-justified-align.svg width="100%" height="100%" %}</div>
# Models: Language Models (TODO)

A language model is a probability distribution over words.

## Representation Learning

Representation learning focuses on encoding text into numerical representations that capture semantic meaning.

### Statistical Methods

- **Bag of Words**: represents text as an unordered collection of words, ignoring grammar and context.
- **Term Frequency Inverse Document Frequency (TF-IDF)**: assigns importance to words based on their frequency in a document relative to their occurrence across a corpus.

### Machine Learning Approaches

- **Word2Vec**: uses neural networks to learn word embeddings. E.g. Continuous Bag of Words (CBOW) and Skip-Gram models.
- **Transformers**: deep learning models that capture long-range dependencies and contextual meaning. E.g. BERT, GPT-4.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs

Reference designs for commonly-asked machine learning system design problems.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Visual Search[^mlsystemdesign]

## Machine Learning Task

- **Objective:** accurately retrieve images visually similar to a query image.
- **Input:** query image.
- **Output:** retrieved images ranked by similarity to the query image.
- **Category:** ranking problem.

---

## Data Preparation

- **Data Engineering:** users, images and user-image interactions.
- **Feature Engineering:** resizing, scaling, normalisation and augmentation.

---

## Model Development

- **Model Selection:** CNN-based models (ResNet) or transformer-based models (ViT).
- **Model Training:** use contrastive learning by training the model to discriminate between similar and dissimilar images. To select the similar image, we can use human judgement, use interaction data as a proxy for similarity, or artificially create a similar image from the query image.
- **Loss Function:** contrastive loss: (i) compute cosine similarities between the query image and the retrieved images, (ii) apply softmax to get probabilities; and (iii) compute the cross-entropy loss between the probabilities and the ground truth labels.

---

## Evaluation

- **Offline Evaluation:** Mean Reciprocal Rank (MRR), Recall@k, Precision@k, mean Average Precision (mAP), normalised Discounted Cumulative Gain (nDCG).
- **Online Evaluation:** Click-through Rate (CTR), average daily/weekly/monthly time spent on suggested images.

---

## Serving

- **Prediction Pipeline:** embedding generation service, nearest neighbour service (exact nearest neighbour or **approximate nearest neighbour**), re-ranking service.
- **Indexing Pipeline:** indexing service.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Google Street View Blurring[^mlsystemdesign]

## Machine Learning Task

- **Objective:** accurately detect privacy-sensitive objects in Street View images so they can be blurred before display.
- **Input:** street View image.
- **Output:** detected objects with bounding boxes and class labels (for example, human face or license plate).
- **Category:** object detection problem, combining **regression** for bounding box coordinates and **classification** for object classes.

---

## Data Preparation

- **Data Engineering:** annotated dataset of 1 million images with labeled bounding boxes for human faces and license plates, plus raw Street View images and metadata such as location, camera pitch/yaw/roll, and timestamp. 
- **Feature Engineering:** standard image preprocessing such as resizing and normalization, followed by image augmentation (random crop, flip, rotation/translation, affine transforms, and brightness/saturation/contrast changes).

---

## Model Development

- **Model Selection:** two-stage network consisting of a region proposal network (RPN) and a classifier (e.g. Fast R-CNN, Faster R-CNN) or one-stage alternatives (e.g. YOLO, SSD) if faster inference becomes necessary.
- **Model Training:** train on annotated images with ground-truth bounding boxes and classes.
- **Loss Function:** regression loss for predicted bounding boxes vs ground-truth boxes and classification loss (for example cross-entropy/log loss) for predicted object classes. Final loss is a weighted combination of classification loss and regression loss.

---

## Evaluation

-  **Offline Evaluation:** Average Precision (AP) per class and mean Average Precision (mAP) across classes, computed using IOU-based correctness thresholds. 
- **Online Evaluation:** user reports/complaints about missed blurs, plus human spot-checking of incorrectly blurred images. 

---

## Serving

- **Prediction Pipeline:** because latency is not critical, the system uses an offline batch prediction pipeline: (i) preprocessing; (ii) blurring service; (iii) non-maximum suppression (NMS) to remove overlapping duplicate detections; (iv) blur detected faces/license plates; and (v) store blurred images for serving.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: YouTube Video Search[^mlsystemdesign]

## Machine Learning Task

- **Objective:** rank videos based on their relevance to a text query.
- **Input:** text query.
- **Output:** retrieved videos ranked by relevance to the query.
- **Category:** ranking problem.

---

## Data Preparation

- **Data Engineering:** annotated dataset of 10 million ⟨video, text query⟩ pairs. 
- **Feature Engineering:** text preprocessing (text normalization, tokenization, token-to-ID conversion) and video preprocessing (decode frames, sample frames, resize, scale, normalize, and correct color mode).

---

## Model Development

- **Model Selection:** 
  - **Text encoder:** statistical methods (bag of words, TF-IDF) or machine learning methods (word2vec, transformers).
  - **Video encoder:** video-level models or frame-level models (ViT). 
- **Model Training:** use contrastive learning to bring matched video-query pairs closer in embedding space and push unmatched pairs apart.
- **Loss Function:** contrastive loss

---

## Evaluation

- **Offline Evaluation:** Mean Reciprocal Rank (MRR), Recall@k, Precision@k, mean Average Precision (mAP), normalised Discounted Cumulative Gain (nDCG).
- **Online Evaluation:** Click-through Rate (CTR), video completion rate, total watch time of search results. 

---

## Serving

- **Prediction Pipeline**
  - **Visual search:** encode query text, retrieve nearest video embeddings using Approximate Nearest Neighbour (ANN).
  - **Text search:** use Elasticsearch over titles, manual tags, and auto-generated tags.
  - **Fusing layer:** combine visual-search and text-search results using a weighted sum of relevance scores.
  - **Re-ranking service:** apply business logic and policies before returning final results. 
- **Indexing Pipelines**
  - **Video indexing pipeline:** compute and store video embeddings for nearest-neighbour retrieval.
  - **Text indexing pipeline:** index titles, manual tags, and auto-generated tags for full-text retrieval. 

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Harmful Content Detection[^mlsystemdesign]

## Machine Learning Task

- **Objective:** accurately predict whether a post contains harmful content so the platform can remove or demote it.
- **Input:** a post containing text, image, video, author information, and user reactions.
- **Output:** harm probability and per-class harmful content probabilities.
- **Category:** multi-modal multi-task classification problem.

---

## Data Preparation

- **Data Engineering:** users, posts, and user-post interactions such as likes, comments, shares, reports, and appeals.
- **Feature Engineering** 
  - **Text features:** preprocess text and encode it with a multilingual pretrained model such as DistilBERT.
  - **Image/video features:** preprocess and extract embeddings using pretrained image/video models.
  - **Reaction features:** counts of likes, shares, comments, reports, plus aggregated comment embeddings.
  - **Author features:** prior violations, total reports, profane-word rate, demographics, follower/following counts, account age.
  - **Contextual features:** time of day and device type. 

---

## Model Development

- **Model Selection:** neural-network-based multi-task classifier with shared layers for common feature transformation and task-specific heads for harmful classes such as violence, nudity, and hate.
- **Model Training:** build fused features offline and store them in a feature store: use natural labels from user reports for training data; and use hand-labeled data for evaluation data.
- **Loss Function:** binary classification loss such as cross-entropy for each task, then combine task-specific losses into an overall loss.

---

## Evaluation

- **Offline Evaluation:** PR-AUC, ROC-AUC.
- **Online Evaluation:** harmful impressions, valid appeals, proactive rate, user reports per harmful class, prevalence.

---

## Serving

- **Prediction Pipeline:** harmful content detection service predicts harmful probabilities for new posts.
- **Enforcement Pipeline:** high-confidence harmful posts should be sent to the violation enforcement service for immediate removal and user notification; while low-confidence harmful posts should be sent to the demoting service, temporarily downranked, and queued for manual review.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Video Recommendation System[^mlsystemdesign]

## Machine Learning Task

- **Objective:** maximize the number of relevant videos.
- **Input:** user.
- **Output:** ranked list of videos sorted by their relevance scores.
- **Category:** recommendation system.

---

## Data Preparation

- **Data Engineering** 
  - **Videos:** ID, length, duration, tags, title, likes, views, language.
  - **Users:** ID, username, age, gender, city, country, language, time zone.
  - **User-video interactions:** likes, clicks, impressions, past searches.
- **Feature Engineering:** 
  - **Contextual information:** time of day, device, day of week.
  - **User histoical interactions:** past searches, liked videos, watched videos, impressions.

---

## Model Development

- **Model Selection:** hybrid filtering: (i) collaborative filtering for candidate generation; and (ii) content-based filtering for later scoring.
- **Model Training:** construct positive user-video pairs if the user liked the video or watched at least half of it and negative user-video pairs from random irrelevant videos or explicitly disliked videos.
- **Loss Function:** for the two-tower model, use cross-entropy loss on binary labels and for matrix factorization, use a weighted squared-loss objective over observed and unobserved entries, optimized with WALS.

---

## Evaluation

- **Offline Evaluation:** Precision@k, mean Average Precision (mAP), diversity.
- **Online Evaluation:** Click-through Rate (CTR), number of completed videos, total watch time, explicit user feedback.

---

## Serving

- **Prediction Pipeline**
  - **Candidate generation:** narrow billions of videos to thousands using ANN over video embeddings.
  - **Scoring:** rank candidates using a richer model and video features.
  - **Re-ranking:** apply additional criteria such as freshness, duplicates, region restrictions, misinformation checks, fairness, and clickbait filtering.
- **Cold Start Handling:** new users rely on demographic/contextual features with two-tower networks; new videos use metadata/content features and heuristics to gather initial interactions before fine-tuning.
</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Event Recommendation System[^mlsystemdesign]

## Machine Learning Task

- **Objective:** maximize the number of event registrations to increase ticket sales.
- **Input:** user.
- **Output:** top-k events ranked by relevance to the user.
- **Category:** ranking problem, implemented as a pointwise learning-to-rank setup with binary classification. 

---

## Data Preparation

- **Data Engineering:** users, events, friendships, and user-event interactions such as impressions, registrations, and invitations. 
- **Feature Engineering:**
  - **Location-related features:** walk/transit/bike scores, same city/country, distance to event, distance similarity.
  - **Time-related features:** remaining time until event, travel time, day/hour preference profiles, day/hour similarity.
  - **Social-related features:** number of registered users, registration ratio, number of registered friends, invitation counts, whether host is a friend, prior registrations with this host.
  - **User-related features:** age and gender.
  - **Event-related features:** price, price similarity, event-description similarity. 

---

## Model Development

- **Model Selection**
  - **Logistic regression:** uses a linear combination of features to predict the probability of an event being registered.
    - Pros: efficient training, fast inference, works well with linearly separable data, interpretable.
    - Cons: can't solve non-linear problems, cannot learn multicollinear features.
  - **Decision tree:** uses a tree-based model to predict the probability of an event being registered.
    - Pros: fast training, fast inference, no data preparation, interpretable.
    - Cons: non-optimal decision boundary, overfits easily.
  - **Gradient-boosted decision tree (GBDT):** uses a tree-based model to predict the probability of an event being registered by iteratively adding weak learners to the model.
    - Pros: easy data preparation, reduces variance, reduces bias, works well with structured data.
    - Cons: many hyperparameters, performs poorly on unstructured data, unsuitable for continual learning.
  - **Neural network (NN):** uses a neural network to predict the probability of an event being registered.
    - Pros: continual learning, works well with unstructured data, expressive.
    - Cons: expensive to train, quality of input is important, large training dataset is required, black-box.
- **Model Training:**
  - Construct training examples from user-event pairs.
  - Label as 1 if the user registered for the event, 0 otherwise.
  - Handle class imbalance using methods such as focal loss, class-balanced loss, or undersampling the majority class. 
- **Loss Function:** binary cross-entropy for the neural network formulation. 

---

## Evaluation

- **Offline Evaluation:** mean Average Precision (mAP), since relevance is binary and ranking quality matters. 
- **Online Evaluation:** click-through rate (CTR), conversion rate, bookmark rate, revenue lift. 

---

## Serving

- **Online Learning Pipeline:** continuously construct datasets from fresh interaction data, train/fine-tune models, evaluate them, and deploy updated models because events are constantly changing and expiring. 
- **Prediction Pipeline**
  - **Event filtering:** narrow down from many events to a candidate set using simple rules such as location and user-selected filters.
  - **Ranking service:** compute features for each ⟨user, event⟩ pair, score candidates with the trained model, and return the top-k events.
  - **Feature computation:** use static features from a feature store and dynamic features computed in real time from raw data. 
</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Ad Click Prediction on Social Platforms[^mlsystemdesign]

## Machine Learning Task

- **Objective:** predict which ads a user is most likely to click, in order to maximize revenue.
- **Input:** user.
- **Output:** ads ranked by click probability.
- **Category:** ranking problem, implemented as a pointwise learning-to-rank setup with binary classification on ⟨user, ad⟩ pairs. 

---

## Data Preparation

- **Data Engineering:** Ads, users, and user-ad interactions such as impressions, clicks, conversions, and hide actions. 
- **Feature Engineering:**
  - **Ad features:** ad ID, advertiser ID, campaign ID, ad group ID, image/video embeddings, category/subcategory, impression counts, click counts, advertiser/campaign engagement stats.
  - **User features:** demographics, contextual information, clicked ads, historical engagement statistics such as ad views and click rate.
  - **Representation methods:** embedding layers for sparse IDs, pretrained image/video encoders for creatives, text processing for category/subcategory, scaling for numeric engagement features. 

---

## Model Development

- **Model Selection:**
Yes — here’s a cleaned-up version with a **one-line description**, plus **pros and cons** for each model section from your notes. Based on the attached text. 

- **Model Selection:**
  - **Logistic regression:** a simple linear binary classifier that predicts click probability from a weighted sum of features.
    - Pros: fast to train, easy to implement, strong baseline.
    - Cons: cannot model non-linear patterns and does not capture feature interactions. 
  - **Feature crossing + logistic regression:** a manually engineered extension of logistic regression that adds crossed features to capture some pairwise interactions.
    - Pros: captures some second-order feature interactions, improves over plain LR when good crosses are chosen.
    - Cons: manual and domain-knowledge-heavy, increases sparsity, and still misses more complex interactions. 
  - **Gradient-boosted decision trees (GBDT):** a tree-ensemble model that learns non-linear decision boundaries and feature importance from structured data.
    - Pros: interpretable and easy to understand.
    - Cons: poor for continual learning and cannot train embedding layers for sparse categorical features. 
  - **GBDT + logistic regression:** a hybrid approach where GBDT first creates/selects stronger features and LR uses them for final click prediction.
    - Pros: GBDT-generated features are often more predictive and easier for LR to learn from.
    - Cons: still weak at capturing complex interactions and too slow for fast continual learning. 
  - **Neural network (single NN):** a deep model that takes the original feature vector directly and predicts click probability end to end.
    - Pros: flexible, expressive, and able to learn non-linear patterns.
    - Cons: struggles with very sparse high-dimensional inputs and has difficulty learning all pairwise interactions efficiently in this setting. 
  - **Neural network (two-tower):** a dual-encoder architecture that separately embeds user features and ad features, then scores them by similarity.
    - Pros: learns user and ad representations explicitly and is useful for matching-style problems.
    - Cons: still suffers from sparse-feature issues here and is not ideal for capturing the full set of useful feature interactions in ad click prediction. 
  - **Deep & Cross Network (DCN):** a model that combines a deep network with a cross network to automatically learn useful feature crosses.
    - Pros: automatically captures feature interactions and is more effective than plain neural networks for this problem.
    - Cons: the cross network only models certain interaction patterns, so performance can still be limited. 
  - **Factorization Machines (FM):** an embedding-based model that extends linear models by explicitly learning all pairwise feature interactions.
    - Pros: efficient for sparse data and very good at modeling pairwise feature interactions.
    - Cons: cannot learn richer higher-order interactions as well as deep neural networks. 
  - **Deep Factorization Machines (DeepFM):** a hybrid model that combines FM for low-order interactions with a DNN for higher-order feature learning.
    - Pros: captures both pairwise and higher-order interactions, making it a strong choice for sparse recommendation/CTR tasks.
    - Cons: more complex and heavier than simpler models, and adding extra stages like GBDT can hurt training, inference, and continual learning speed. 
- **Model Training:**
  - Construct one data point per ad impression, labelling it positive if the user clicks within time threshold (t) and negative otherwise. 
- **Loss Function:** cross-entropy for binary classification. 

---

## Evaluation

- **Offline Evaluation:** cross-entropy (CE), normalized cross-entropy (NCE). 
- **Online Evaluation:** click-through rate (CTR), conversion rate, revenue lift, hide rate. 

---

## Serving

- **Data Preparation Pipeline:** compute online and batch features and continuously generate training data from new ads and interactions.
- **Continual Learning Pipeline:** Continuously fine-tune, validate, and deploy updated models because even short delays in learning from fresh data hurt performance. 
- **Prediction Pipeline:** filter the pool of ads using advertiser targeting criteria such as age, gender, and country, compute click probabilities for candidate ads using static and dynamic features, and apply business logic such as increasing diversity and removing overly similar ads. 
</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Similar Listings on Vacation Rental Platforms[^mlsystemdesign]

## Machine Learning Task

- **Objective:** accurately predict which listing a user will click next, given the listing they are currently viewing, in order to increase bookings.
- **Input:** currently viewed listing.
- **Output:** ranked list of similar listings.
- **Category:** session-based recommendation problem using listing embeddings. 

---

## Data Preparation

- **Data Engineering:** users, listings, and user-listing interactions such as impressions, clicks, and bookings. 
- **Feature Engineering:** search sessions, where each session is a sequence of clicked listing IDs followed by an eventually booked listing. 

---

## Model Development

- **Model Selection:** a shallow neural network to learn listing embeddings. 
- **Model Training:** learn embeddings from co-occurrences of listings in browsing sessions using a sliding window, with positive pairs from a central listing and its context listings and negative pairs from randomly sampled listings using negative sampling.
- **Loss Function:** use cross-entropy loss with the eventually booked listing as a global positive context and hard negatives from the same region.

---

## Evaluation

- **Offline Evaluation:** average rank of the eventually booked listing, measured by ranking listings relative to the first clicked listing in each session. 
- **Online Evaluation:** click-through rate (CTR), session book rate. 

---

## Serving

- **Training Pipeline:** fine-tune the model regularly using new listings and fresh user-listing interactions. 
- **Indexing Pipeline:** precompute listing embeddings for all listings and store them in an index table; refresh the table when new listings or models arrive. 
- **Prediction Pipeline:** fetch embedding for the current listing; use heuristics for unseen new listings; use approximate nearest neighbor (ANN) search to retrieve similar listings efficiently; apply user filters and constraints such as city or price filters before display.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Personalized News Feed[^mlsystemdesign]

## Machine Learning Task

- **Objective:** maximize a weighted engagement score of implicit reactions (such as dwell time or user clicks) and explicit reactions (such as likes, shares, or hides).
- **Input:** user.
- **Output:** ranked list of unseen posts sorted by engagement score.
- **Category:** ranking problem, implemented as a pointwise learning-to-rank setup with multi-task prediction of reactions. 

---

## Data Preparation

- **Data Engineering:** users, posts, user-post interactions, and friendship data. 
- **Feature Engineering:**
  - **Post features:** textual content, images/videos, reaction counts, hashtags, post age.
  - **User features:** demographics, contextual information, user-post interaction history, whether the user is mentioned.
  - **User-author affinity features:** like/click/comment/share rates with the author, friendship length, close-friend/family indicators. 
- **Representation Methods:** pretrained BERT for post text, pretrained image/video models such as ResNet or CLIP, lighter text methods such as TF-IDF or word2vec for hashtags, scaling for numeric counts, bucketization and one-hot encoding for post age and other categorical/time features. 

---

## Model Development

- **Model Selection:** multi-task DNN instead of $$n$$ independent DNNs, so the model can share representations across reactions and better handle sparse reactions. 
- **Model Extension for Passive Users:** add two implicit reactions to the list of tasks: dwell-time prediction and skip prediction.
- **Model Training:** create positive examples from observed reactions such as likes, shares, or comments, create negative examples from impressions without that reaction, balance positive and negative samples for each binary reaction task, and for dwell time, use impression records with dwell-time labels directly.
- **Loss Function:** binary cross-entropy for binary reaction tasks and a regression loss such as MAE, MSE, or Huber loss for dwell time. 

---

## Evaluation

- **Offline Evaluation:** ROC-AUC for each binary reaction classifier. 
- **Online Evaluation:** click-through rate (CTR), reaction rate, total time spent, user satisfaction rate from surveys.

---

## Serving

- **Data Preparation Pipeline:** see [Designs: Ad Click Prediction on Social Platforms](#designs-ad-click-prediction-on-social-platforms)
- **Prediction Pipeline:**
  - **Retrieval service:** retrieves posts that a user has not seen, or which have comments also unseen by them.
  - **Ranking service:** ranks the retrieved posts by assigning an engagement score to each one.
  - **Re-ranking service:** modifies the list of posts by incorporating additional logic and user filters.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: People You May Know[^mlsystemdesign]

## Machine Learning Task

- **Objective:** maximize the number of formed connections between users.
- **Input:** user.
- **Output:** ranked list of potential connections.
- **Category:** edge prediction on a social graph, used for ranking candidate users. 

---

## Data Preparation

- **Data Engineering:** users, connections, and interaction data such as connection requests, accepted requests, searches, profile views, and reactions. 
- **Data Standardization:** normalize educational/work fields such as school, degree, major, company, and industry, since the same attribute may appear in different textual forms. 
- **Feature Engineering:**
  - **User features:** demographics, number of connections/followers/following/pending requests, account age, received reactions.
  - **User-user affinity features:** schools in common, overlapping school years, same major, companies in common, same industry.
  - **Social affinity features:** profile visits, number of mutual connections, and time-discounted mutual connections. 

---

## Model Development

- **Model Selection:** graph neural network (GNN), since it can process the social graph directly and produce node embeddings for users. Possible architectures mentioned include GCN, GraphSAGE, GAT, and GIT. 
- **Model Training:**
  - Create a snapshot of the social graph at time (t).
  - Compute initial node features from user attributes.
  - Compute initial edge features from affinity features.
  - Use the graph at time (t+1) to label whether a new edge forms between user pairs. 
- **Loss Function:** beyond the scope of this design.

---

## Evaluation

- **Offline Evaluation:** ROC-AUC for the GNN edge-prediction model and mAP for the PYMK ranking system. 
- **Online Evaluation:** total number of connection requests sent, total number of connection requests accepted. 

---

## Serving

- **PYMK Generation Pipeline:**
  - **FoF service:** generates candidate connections from 2-hop neighbors instead of the full user graph to reduce the search space.
  - **Scoring service:** score candidates with the GNN model, storing precomputed PYMK results in a database because the social graph changes relatively slowly.
- **Prediction Pipeline:** fetch precomputed recommendations directly when available; if missing, issue a one-time request to the generation pipeline.

</section>

<section class="relative mb-4 sm:mb-8 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-950/5 p-4 dark:bg-white/5" markdown="1">
<div class="absolute -top-2 right-4 size-16 text-zinc-200 dark:text-zinc-700">{% svg /assets/images/streamline/hierarchy-4.svg width="100%" height="100%" %}</div>
# Designs: Customer Support Agent

## Machine Learning Task

- **Objective:** increase resolution rate and customer satisfaction while reducing human-agent cost.
- **Input:** user query plus multi-turn conversation context.
- **Output:** factual customer-service response, clarification question, recommended action, or SOP-guided solution.
- **Category:** multi-stage agentic conversational system combining retrieval, intent understanding, policy/action decision, and response generation. 

---

## Data Preparation

- **Data Engineering:** Business corpus, SOP workflow documents, FAQ entries, policy rules, troubleshooting guides, historical customer-service conversations, online good cases, online bad cases, and synthetic trajectories from self-play. 
- **Knowledge Engineering:**

  - **Knowledge types:** FAQs, general knowledge, and SOP solutions.
  - **Knowledge operations:** freshness updates for new campaigns/policies and gap addition from unmatched queries.
  - **Retrieval assets:** indexed and chunked documents for hybrid search and reranking. 
- **Training Data Construction:**
  - synthetic multi-turn conversations from user simulator and assistant self-play;
  - successful golden trajectories for supervised fine-tuning;
  - flagged bad online cases corrected for learning;
  - query-document relevance annotations;
  - intent-annotated conversational datasets;
  - intent-document-action and query-document-response triplets. 

---

## Model Development

- **Model Selection:** finetuned models for domain comprehension and serving efficiency, retrieval agent for RAG, master agent following the PEER paradigm: Plan → Execution → Expression → Reflection. 
- **Core Components:**
  - **Knowledge Retrieval:** query rewriting, hybrid search (dense embeddings + sparse BM25), reranking.
  - **Intent Understanding:** multi-round context-aware intent reconstruction and clarification.
  - **Policy Decision:** decide whether to clarify, answer directly, call tools, or delegate to subagents.
  - **Response Generation:** grounded, empathetic, professional replies with recommended actions/buttons. 
- **Training Pipeline:**
  - **CPT:** continuous pre-training for domain adaptation and terminology alignment.
  - **SFT:** supervised fine-tuning on golden trajectories.
  - **Agentic RL:** GRPO-based reinforcement learning for preference alignment and goal achievement optimization. 
- **Loss / Optimization Objective:** system is isolated by component and optimized through a combination of supervised learning and agentic RL using reward signals for task success, factualness, conversation consistency, and style preference.

---

## Evaluation

- **Offline Evaluation:** relevance ranking quality, intent classification accuracy, policy accuracy, response correctness, groundedness / factual accuracy, empathy, goal achievement score, conversation consistency, style preference. 
- **Online Evaluation:** resolution rate (RR), customer satisfaction (CSAT), response time, hallucination/safety performance. 

---

## Serving

- **Prediction Pipeline:**
  - **Safety & Routing:** zero-shot classifier routes to safety refusal, business inquiry, human service, or chitchat.
  - **Retrieval Agent:** retrieves grounding knowledge with query rewriting, hybrid search, and reranking.
  - **Master Agent:** performs multi-round intent understanding and PEER-based planning/execution.
  - **Tool Calls:** invoke SOP agent, databases, external APIs, calculators, or other integrations as needed.
  - **Response Generation:** produce final factual, empathetic, professional response or guided action. 
- **Platform Architecture:**
  - application layer for customer-service assistant and AI search;
  - agent/tools layer for workflow orchestration, MCP tools, memory, and evaluation;
  - model/data layer for inference/training, ETL, analytics, and knowledge-base management; and
  - infrastructure layer for compute plus security/compliance. 
- **Continuous Improvement Loop:** monitor RR/CSAT/latency, review good/bad cases weekly, refine models, update knowledge base for freshness and coverage gaps.
  
</section>

[^cs229]: [CS3229: Machine Learning](https://cs229.stanford.edu/)

[^cs224n]: [CS3224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)

[^cs231n]: [CS3314N: Deep Learning for Computer Vision](http://vision.stanford.edu/teaching/cs231n/)

[^cs324]: [CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/)

[^cs329s]: [CS329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/)

[^yan24]: [Task-Specific LLM Evals that Do & Don’t Work](https://eugeneyan.com/writing/evals/)

[^esl]: [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

[^huyen22]: [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

[^isl]: [An Introduction to Statistical Learning](https://www.statlearning.com/)

[^mlinterviews]: [Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/)

[^mlsystemdesign]: [Machine Learning System Design Interview](https://bytebytego.com/intro/machine-learning-system-design-interview)

[^coursera]: [Coursera Deep Learning Specialisation](https://www.coursera.org/specializations/deep-learning)

[^data_checklist]: [Is My Data Any Good? A Pre-ML Checklist.](https://services.google.com/fh/files/blogs/data-prep-checklist-ml-bd-wp-v2.pdf)

[^generativeai]: [Google Generative AI Learning Path](https://www.cloudskillsboost.google/paths/118)

[^google]: [Google Machine Learning Education](https://developers.google.com/machine-learning)

[^hugging_face]: [Hugging Face Documentation](https://huggingface.co/docs)

[^intro_reinforcement_learning]: [Introduction to Reinforcement Learning](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)

[^ml_rules]: [Rules of Machine Learning: Best Practices for ML Engineering.](https://developers.google.com/machine-learning/guides/rules-of-ml)

[^pair]: [People + AI Guidebook](https://pair.withgoogle.com/guidebook/)

[^pandas]: [Pandas Documentation](https://pandas.pydata.org/docs/)

[^pytorch]: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

[^pytorch_lightning]: [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

[^reinforcement_learning]: [Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)

[^sklearn]: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

[^spinning_up]: [Spinning Up in Deep Reinforcement Learning](https://spinningup.openai.com/en/latest/)
