# Linear Text Classification

## Linear Text Classification vs Nonlinear Text Classification: 

Linear classifiers are simpler, faster, and more interpretable, making them suitable for straightforward tasks or when computational resources are limited. Nonlinear classifiers, on the other hand, are more powerful and versatile, ideal for complex tasks but require more data, computational power, and expertise to develop and interpret.

### Linear Text Classification:

- Concept: This involves classifying text into categories based on linear algorithms. The most defining feature is that it assumes a linear relationship between the input features (like words or phrases in the text) and the output categories.

- Algorithms: Common algorithms include Logistic Regression, Naive Bayes, and Linear Support Vector Machines (SVM). These algorithms work well with high-dimensional sparse data, which is typical in text classification.

- Applications: Linear classifiers are often used for simpler or well-defined tasks such as spam detection, sentiment analysis, or categorizing news articles into predefined topics.

- Advantages: They are generally faster to train and easier to interpret. Also, they require less computational resources compared to nonlinear models.

### Nonlinear Text Classification:

- Concept: Nonlinear classification doesn't assume a linear relationship between input and output. It's capable of capturing more complex patterns in data, which can be crucial for intricate text classification tasks.

- Algorithms: Commonly used algorithms include Decision Trees, Random Forest, Neural Networks, and Deep Learning models like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

- Applications: These classifiers are ideal for complex tasks like language translation, emotion analysis, and other tasks where context and the sequence of words are important.

- Advantages: Nonlinear classifiers can achieve higher accuracy in complex scenarios and are better at handling the intricacies of human language. They are particularly effective in large datasets with complex relationships.


---
### Bag-Of-Words:

The "Bag of Words" (BoW) model is a simple and commonly used way in natural language processing to represent text data. In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. The Bag of Words model is powerful due to its simplicity and efficiency in converting text into numerical data, which can then be used for various computational processes like classification or clustering in machine learning. However, it has limitations such as not capturing the order of words and the context they are used in.

While the Bag of Words model is a useful starting point in many NLP tasks due to its simplicity and efficiency, its limitations in handling the complexity and nuances of human language mean that it is often supplemented or replaced by more advanced techniques in more sophisticated NLP applications.

##### Summary of the Bag of Words model with an Example:

"The cat sat on the mat.
The dog sat on the log."

1. List Unique Words: It starts by identifying all the unique words in the text. -> the, cat, sat, on, mat, dog, log

2. Create a Vector: For each unique word, create a vector with a place for every unique word. The length of this vector is equal to the number of unique words.

3. Count the Words: In the vector, count how many times each word appears in the text. This count is placed in the corresponding position in the vector.


- Vocabulary Unique Vector --> {the, cat, sat, on, mat, dog, log}

- The count of each of these words in the sentences is reflected in the vector.
- "The cat sat on the mat" --> [2, 1, 1, 1, 1, 0, 0]
- "The dog sat on the log" --> [2, 0, 1, 1, 0, 1, 1]

   


#### Advantages:

- Simplicity: BoW is straightforward to understand and implement, making it accessible for beginners in NLP and suitable for baseline models.

- Efficiency: It is computationally efficient, especially for large datasets, as it involves simple counting of word occurrences.

- Flexibility: BoW can be easily combined with other algorithms and techniques for more complex NLP tasks like text classification, clustering, and sentiment analysis.

- Document Comparison: It allows for a basic form of document comparison and retrieval by transforming text into a numerical form that can be easily compared and processed.

- Good for Text Classification: BoW, especially when combined with techniques like TF-IDF (Term Frequency-Inverse Document Frequency), can be quite effective in tasks like spam filtering, sentiment analysis, and topic modeling.


#### Disadvantages:

- Loss of Context and Order: BoW ignores the order of words and their context within the text. This means it can miss the nuances in language such as sarcasm, negation, and idiomatic expressions.

- Sparsity: The BoW representation can become very sparse, especially with large vocabularies, as most words do not appear in each document. This can lead to inefficiency in storage and computation.

- Vocabulary Size: The size of the vocabulary can become a limitation, as it grows with the number of unique words in the dataset, potentially leading to high dimensionality.

- Word Ambiguity: BoW does not handle word ambiguity well. Different meanings of the same word are treated identically because it doesn’t take into account the context.

- Frequency Bias: Common words that appear frequently in all documents (like 'the', 'is', 'and') can dominate in the representation, overshadowing rare but potentially more meaningful words.

- Ignoring Semantics: The model does not capture the semantic relationships between words, such as synonyms or antonyms, leading to a lack of depth in understanding the text.


---

### Weights to use Linear Text classification:


#### Naive Bayes Method:


![Screenshot 2024-01-13 at 12 11 49 PM](https://github.com/kasteway/Natural_Language_Processing/assets/62068733/7edc0503-78c5-404a-8798-fcd493ecd6b3)


### Perceptron Method:

![Screenshot 2024-01-13 at 12 12 17 PM](https://github.com/kasteway/Natural_Language_Processing/assets/62068733/20b59985-b49c-4113-bab9-0a6e46e11c53)
![Screenshot 2024-01-13 at 12 12 30 PM](https://github.com/kasteway/Natural_Language_Processing/assets/62068733/42ecd190-60eb-4524-bd48-71161f0972d6)



---

### Predicting a label from Bag-of-Words:


![Screenshot 2024-01-13 at 12 11 04 PM](https://github.com/kasteway/Natural_Language_Processing/assets/62068733/10527d6b-0f73-4692-80c7-490a9f1d2771)


![Screenshot 2024-01-13 at 12 09 43 PM](https://github.com/kasteway/Natural_Language_Processing/assets/62068733/f8c838fb-201c-4298-8a2d-8c74533a061b)


