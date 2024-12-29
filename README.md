# Spam-Email-Detection
The Spam Email Detection project uses machine learning to identify and filter spam emails from legitimate ones (ham). By analyzing the content and structure of emails, the system learns patterns and features commonly found in spam, such as specific keywords, links, or sender information.
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
  
  &emsp;&emsp;&emsp;**Encoder Layer** :The encoder consists of multiple layers of attention mechanisms and feed-forward neural networks.</br>
  
   &emsp; &emsp;&emsp; &emsp;_Self-Attention_ : Each word or token in the email attends to every other token in the email to capture contextual relationships.</br>
   
  &emsp; &emsp; &emsp;&emsp;_Multi-Head Attention_ : he model uses multiple attention heads to capture different aspects of the relationships between words.</br>
  
  &emsp; &emsp; &emsp;&emsp;_Feed-Forward Layer_ : A feed-forward neural network processes the information. The output of this layer is then passed through a normalization layer.</br>
  &emsp; &emsp; &emsp;_Layer Normalization_ : Normalization is used to stabilize the training process by scaling and shifting the outputs of each layer.</br>
### **Model Characteristics :**</br>

&emsp;_num_heads_: Number of parallel attention mechanisms used in a multi-head attention layer.</br>

&emsp;_vocab_size_: The total number of unique tokens in the model's vocabulary.</br>

&emsp;_embed_dim_: The size of the vector representation for each token in the input sequence.</br>

&emsp;_ff_dim_: The dimensionality of the inner layer in the feedforward neural network used in the Transformer model.</br>

&emsp;_max_seq_len_: The maximum length of input sequences the model can process.</br>

### **Model Dimensions :**

&emsp;_Number of Labels_: 2 (Spam vs. Ham)</br>

&emsp;_Input Sequence Length_: Up to 512 tokens.</br>

&emsp;_Number of Transformer Layers_: 6 layers.</br>

&emsp;_Number of Attention Heads per Layer_: 12 heads.</br>

&emsp;_Intermediate Feedforward Layer Size_: 3072.</br>
### **Model Output :**
&emsp;_Classification head_ : This layer takes the output from the Transformer and produces a single probability score representing the likelihood of the email being spam or ham.</br>
&emsp;_Softmax Layer :_ Converts scores into probabilities, showing how likely the email is to belong to each class.
## TRAINING FUNCTION
### **Training Parameters :**
&emsp;_Number of Training Epochs:_ Specifies how many complete passes through the training dataset the model will perform.</br>

&emsp;_Training Batch Size:_ Determines the number of samples processed together during a forward and backward pass. </br>

&emsp;_Evaluation Batch Size:_ Optimized for faster inference while maintaining memory efficiency.</br>

&emsp;_Learning Rate:_ Defines the step size at which the optimizer updates the model weights.</br>

## **Model Training :**

1. **_Input Preparation :_** Input consists of tokenized sequences (texts) with their binary labels (spam: 1, ham: 0).</br>

2. **_Forward Pass :_** The tokenized input is passed through the Transformer model, which outputs logits for binary classification.
Loss Calculation: The Binary Cross-Entropy Loss with Logits is calculated.</br>

3. **_Backpropagation :_** Gradients are calculated.</br>

4. **_Optimization :_** The optimizer updates the model's parameters.</br>

5. **_Validation :_** The model is evaluated on the validation set.</br>

## **Training Function Parameters:**

**_Loss Function :_** torch.nn.BCEWithLogitsLoss().</br> It combines a sigmoid layer with binary cross-entropy loss for better numerical stability.</br>

**_Optimizer :_** torch.optim.AdamW with weight_decay=0.01 for regularization.</br>

**_Learning Rate Scheduler :_** Linear decay scheduler with warmup (e.g., get_scheduler from HuggingFace).</br>

## **Evaluation Metrics :**
**_Loss :_** The average cross-entropy loss measures the difference between the predicted probabilities and the actual class labels.</br>

**_Accuracy :_** Percentage of correctly classified examples.</br>

**_Precision :_** Measures the proportion of positive predictions (spam) that were actually correct.</br>

**_Recall :_** Measures the proportion of actual positives (spam) that were correctly identified.</br>

**_F1-Score :_** The harmonic mean of precision and recall, providing a balance between the two. </br>

***F1-score (weighted) :*** Weighted average of F1-scores per class.</br>
