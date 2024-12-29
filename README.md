# Spam-Email-Detection
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
