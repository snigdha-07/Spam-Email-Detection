# Spam-Email-Detection

## DATA
The dataset is obtained from kaggle.</br>
It contains a total of 5572 sample emails among which 4,825 are ham (not spam) and 747 are spam.</br>

### **Initial Exploration And Data Cleaning**
The initial examination of data to identify patterns, trends, anomalies, and potential insights.</br>
The process of correcting missing values, duplicates, and errors in the data.</br>

### **Exploratory Data Analysis (EDA)**
&emsp;1) Distribution of Labels.</br>
&emsp;2) Average Length of Emails for Spam and Ham.</br>
&emsp;3)Average Word of Emails for Spam and Ham.</br>
&emsp;4) Relationship between Length and Spam.</br>
&emsp;5) Relationship between Features.</br>

### **Data Pre-Processing**
&emsp;The basic data pre-processing steps of spam detection are:</br>
&emsp;&emsp;1) Converting the input to lowercase.</br>
&emsp;&emsp;2) Tokenisation.</br>
&emsp;&emsp;3) All the special characters are removed.</br>
&emsp;&emsp;4) Removing stop words and Punctuation.</br>
&emsp;&emsp;5) The word frequency of all the words.</br>
&emsp;&emsp;6) Stemming to reduce words to their root forms.</br>

### **Label Encoding and Vectorization**
&emsp;_Label Encoding_ : A technique used to transform categorical labels (text labels like "ham" and "spam") into numeric labels for machine learning algorithms, which often work better with numerical data.</br>
&emsp;_Vectorization_ : A process to convert textual data into numerical features so that machine learning models can process it. Textual data is unstructured, and vectorization makes it machine-readable.</br>

## MODEL

### **Model Layers :**</br>
  &emsp;_Word and Positional embedding layer_ : Creates a word embedding, which projects the input indexes, (i.e., the tokens), into a vector space, that contains unique and informative representations for each token.</br>
  &emsp;_Transformer Layer_ :The Transformer network processes input text as word embeddings through several Transformer blocks. It generates a vector representation of the text, which is used for classification with a softmax layer. Using self-attention and multi-head attention, the network effectively captures relationships between words, enabling accurate predictions.</br> 
  
### **Model Characteristics :**</br>
&emsp;_num_heads_: Number of parallel attention mechanisms used in a multi-head attention layer.</br>
&emsp;_vocab_size_: The total number of unique tokens in the model's vocabulary.</br>
&emsp;_embed_dim_: The size of the vector representation for each token in the input sequence.</br>
&emsp;_ff_dim_: The dimensionality of the inner layer in the feedforward neural network used in the Transformer model.</br>
&emsp;_max_seq_len_: The maximum length of input sequences the model can process.</br>

### Naïve Bayes Classifier : 
The Naïve Bayes classifier is a supervised machine learning algorithm that is used for classification tasks such as text classification. They use principles of probability to perform classification tasks.</br>

### **Model Dimensions :**</br>
&emsp;_Number of Labels_: 2 (Spam vs. Ham)</br>
&emsp;_Input Sequence Length_: Up to 512 tokens.</br>
&emsp;_Number of Transformer Layers_: 6 layers.</br>
&emsp;_Number of Attention Heads per Layer_: 12 heads.</br>
&emsp;_Intermediate Feedforward Layer Size_: 3072.</br>
