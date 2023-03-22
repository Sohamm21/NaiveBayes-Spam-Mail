# NaiveBayes-Spam-Mail

Naive Bayes is a popular classification algorithm in machine learning that is based on Bayes' theorem. It is called "naive" because it assumes that the features in the dataset are independent of each other, which is not always the case in practice. Despite this simplification, naive Bayes classifiers can be surprisingly accurate for a wide range of classification tasks.

Naive Bayes classifiers work by calculating the probability of each class given a set of input features. The class with the highest probability is then assigned as the predicted class for the input data. The probability calculation is based on Bayes' theorem, which states that the probability of a hypothesis (in this case, a class label) given some observed evidence (in this case, the input features) is proportional to the probability of the evidence given the hypothesis multiplied by the prior probability of the hypothesis.

This is a simple email spam classifier built using the Naive Bayes algorithm in Python. It takes an input email message and predicts whether it is a spam message or not.

### You can install the required packages using pip:
```python
pip install -r requirements.txt
```

### Usage:
First of all you need streamlit environment in your system.
Then, you have to download the **NB_email.py** file and **spam.csv** file in your system.
After cloning, use
```python
streamlit run filename.py
```
to run this file.

## This is how it looks:
![Screenshot of output](https://github.com/Sohamm21/NaiveBayes-Spam-Mail/blob/main/NB.png)
