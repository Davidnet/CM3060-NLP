# CM3060-NLP: Comparative Analysis of News Categorization Models: Naive Bayes vs. GPT-2 with JAX

This project provides a comprehensive comparative analysis between a classical statistical model (Multinomial Naive Bayes) and a modern deep learning model (a 45M parameter GPT-2 variant) for the task of multi-class news categorization. The primary goal is to evaluate the trade-offs in predictive performance, computational cost, and model complexity between these two approaches.

A key objective of this project was to explore and utilize **Google's JAX library**, a high-performance framework for machine learning research. The GPT-2 model was implemented from scratch and trained on a **Tensor Processing Unit (TPU)**, showcasing the power and efficiency of modern hardware accelerators in handling large-scale neural networks.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [Conclusion and Reflections](#conclusion-and-reflections)

## Introduction

### Domain-Specific Area
The global production of text news media has reached an unprecedented volume, with thousands of sources generating content at a gigantic rate. Consequently, manually categorizing text articles is a challenging problem. An automated news categorization system can address this issue, providing a valuable tool for organizing documents. This capability can help both the public and journalists, and could even serve as a core component for custom news aggregators or even for applications in financial analysis.

### Objectives
The aim of this project is to generate a comparative analysis between a classical statistical model and a modern deep learning model for the task of multi-class news categorization. This report will evaluate the trade-offs in predictive performance, between the two approaches. Furthermore, this project serves as an excuse for the exploration of Google's JAX library, a high-performance framework for machine learning research. A key objective is to leverage JAX and the high level library NNX to implement and train the deep learning model, with the goal of potentially using specialized hardware like a Tensor Processing Unit (TPU) to assess the capabilities of this hardware accelerator.

## Dataset

### Description
The dataset used for this project is the AG's news topic classification dataset. The original corpus is a collection of more than one million news articles gathered from over 2,000 news sources by the ComeToMyHead academic news search engine. As described in the TensorFlow Datasets catalog, this collection was assembled over more than a year of activity starting in 2004.

The final benchmark dataset was constructed by Xiang Zhang, Junbo Zhao, and Yann LeCun for their paper, "Character-level Convolutional Networks for Text Classification" (NIPS 2015). They created the dataset by selecting the four largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples, making it a well-balanced dataset for text classification research.

| Property | Description |
| :--- | :--- |
| **Dataset Name** | AG News (Subset) |
| **Source** | TensorFlow Datasets (`ag_news_subset`) |
| **Total Samples** | ~127,600 |
| **No. of Classes** | 4 |
| **Class Labels** | 1 (World), 2 (Sports), 3 (Business), 4 (Sci/Tech) |
| **Features** | Title and Description (Text) |

### Data Splits
The dataset is already split in train, and test with:

| Split | Examples |
|---|---|
| 'test' | 7,600 |
| 'train' | 120,000 |

## Methodology

### Models Used

#### Baseline Model
The baseline model chosen for this project is a **Multinomial Naive Bayes (MNB)** classifier, implemented using scikit-learn. This classifier is highly suitable for text classification tasks involving discrete features like word counts or TF-IDF scores. It serves as a strong and standard baseline, representing a "bag-of-words" approach where the order of words is disregarded.

#### Deep Learning Model
The deep learning model is a **GPT-2 architecture** inspired by the educational implementation detailed by Andrej Karpathy. This project adapts its fundamental design, translating the core logic to JAX to leverage its high-performance computing capabilities. The model consists of 6 transformer blocks, an embedding dimension of 512, and a total of approximately 45 million parameters.

### Evaluation Metrics
The performance of both models is evaluated using the following standard metrics for classification:

| Metric | Description |
| :--- | :--- |
| **Accuracy** | The ratio of correctly predicted instances to the total instances. |
| **Precision** | The ability of the classifier not to label a sample as positive that is actually negative. |
| **Recall** | The ability of the classifier to find all the positive samples. |
| **F1-Score**| The weighted average of Precision and Recall, providing a single score that balances both concerns. |

To provide an overall performance score for the models in this multi-class classification task, the per-class metrics are averaged (using macro-averaging) across all four categories.

## Implementation Details

### Data Preprocessing

#### Statistical Model (Naive Bayes)
For the baseline statistical model, the text was converted into numerical vectors using a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer from scikit-learn. This process involved:
- Tokenization
- Stop word removal
- Limiting the vocabulary to the 10,000 most frequent words.

#### Deep Learning Model (GPT-2)
For the deep learning model, a more sophisticated pipeline was implemented:
1.  The 'title' and 'description' fields from the dataset were concatenated into a single text input.
2.  This text was then tokenized using the standard **"gpt2" tokenizer** from the `tiktoken` library.
3.  To ensure uniform input size, each tokenized sequence was either truncated or padded with the end-of-text (`<|endoftext|>`) token to a fixed length of 256.
4.  This entire process was integrated into a `tf.data` pipeline for efficient batching and processing during training.

### Dependencies
To run the notebook and reproduce the results, the following dependencies are required:

- `matplotlib`
- `seaborn`
- `pandas`
- `tensorflow-datasets`
- `scikit-learn`
- `tiktoken`
- `tpu-info`
- `tensorflow`
- `jax-ai-stack`
- `jax[tpu]`
- `flax`
- `optax`
- `orbax-checkpoint`

## Usage

### Instructions
The entire project is contained within the `CM3060NaturalLanguageProcessingSubmission.ipynb` Jupyter notebook. To run the project, follow these steps:

1.  **Open in Google Colab:** The notebook is designed to be run in Google Colab to take advantage of the free TPU resources. Click the "Open in Colab" badge at the top of the notebook.

2.  **Set Runtime to TPU:** In Colab, navigate to `Runtime > Change runtime type` and select **TPU** as the hardware accelerator.

3.  **Install Dependencies:** The notebook includes cells to pip install all necessary libraries. Run these cells first.

4.  **Execute Cells:** Run the notebook cells sequentially to perform data preprocessing, train the models, and evaluate their performance.

### Model Training
The notebook is configured to train both the Naive Bayes and GPT-2 models. The GPT-2 model is set to train for 5 epochs, which takes approximately 6 minutes on a TPU. The trained model is saved to the `/content/checkpoints/` directory and can be reloaded to avoid retraining.

## Results and Analysis

### Performance Metrics
The following table summarizes the performance of both models on the test set:

| Metric | Statistical Model (Naive Bayes) | Deep Learning Model (JAX GPT-2) |
| :--- | :--- | :--- |
| Accuracy | 0.8879 | 0.9163 |
| Macro Avg Precision | 0.8875 | 0.9162 |
| Macro Avg Recall | 0.8879 | 0.9163 |
| Macro Avg F1-Score | 0.8876 | 0.9162 |
| Training Time | ~30 seconds | ~6 minutes |

### Comparative Discussion
As shown in the table, the JAX GPT-2 model surpasses the Naive Bayes baseline across all metrics. However, the performance gain of approximately 3-4% might initially seem modest given the vast difference in model size and complexity. This highlights a key finding: for a well-structured dataset like AG News, the marginal returns of a much larger model can be subtle when looking only at top-line metrics.

A more insightful analysis comes from the confusion matrices. While the baseline model performs well, the GPT-2 model demonstrates a superior ability to distinguish between classes. The most notable area of confusion for the advanced model is between the 'Business' and 'Sci/Tech' categories. This is an expected and understandable difficulty, as news articles in these domains often share significant thematic and overlap (e.g., a story about a tech company's quarterly earnings). The model's ability to separate these closely related topics, even if imperfectly, showcases its more nuanced understanding of the text compared to the baseline.

## Conclusion and Reflections

### Project Summary
This project successfully conducted a comparative analysis of a classical statistical model (Multinomial Naive Bayes) and a modern deep learning model (a 45M parameter GPT-2 variant) for news classification. The results confirmed the superior performance of the deep learning approach, which achieved an F1-score of 0.9162 compared to the baseline's 0.8876. This validates the hypothesis that sequence-aware models which process contextual information provide a distinct advantage over simpler "bag-of-words" methods.

A central objective was to gain practical experience with Google's JAX library and its high-level NNX interface for training on specialized hardware. This goal was unequivocally met. The implementation of a GPT-2 model from scratch provided a deep, hands-on understanding of the Transformer architecture. Leveraging JAX's Just-In-Time (JIT) compilation on a TPU demonstrated its profound efficiency, enabling the training of a large model in under six minutes—a task that would be significantly slower on conventional hardware. This experience highlighted the power and scalability of modern ML frameworks for tackling large-scale problems.

### Future Work
For future work, several avenues could be explored to build upon these results:
- **Fine-tuning a pre-trained model:** Fine-tuning a pre-trained GPT-2 model, rather than training from scratch, would likely yield a significant performance boost by leveraging existing linguistic knowledge.
- **Hyperparameter tuning:** The current model could be further optimized through hyperparameter tuning.
- **Training on a larger corpus:** While the AG News dataset is substantial, training on an even larger and more diverse corpus could enhance the model's generalization capabilities.
