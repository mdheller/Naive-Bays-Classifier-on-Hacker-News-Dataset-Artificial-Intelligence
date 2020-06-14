
## Naive Bays Classifier on Hacker News Dataset Artificial Intelligence

**PREREQUISITES** 

    IDE      : Jupyter Notebooks ( Download & Install : https://jupyter.readthedocs.io/en/latest/install.html )
    LANGUAGE : Python            ( Download & Install : https://www.python.org/downloads/)

**Libraries Used**   ( NOTE : if library not installed in device, Install it using : pip install <library_name> )
     
    pandas, numpy, matplotlib,math, re, sys, os, string

**How to Run**
      
    1. Put Naive_Bays_Classifier.ipynb, dataset.csv and stopwords.txt in the same folder.
    2. Run Naive_Bays_Classifier.ipynb => Goto Cell->Run All

**Output Files**
    
    for Task 1, 2 : baseline-result.txt,  model-2018.txt,  vocabulary.txt,  remove_word.txt
    for Task 3    : stopword-model.txt,  stopword-result.txt,  wordlength-model.txt,  wordlength-result.txt
        

## **Description**

Hacker News is a popular technology site, where user-submitted stories (known as "posts") are voted and commented upon. The site is extremely popular in technology and start-up circles. The top posts can attract hundreds of thousands of visitors. This dataset contains Hacker News posts from 2018- to 2019 and each post includes the following columns:
  
    Object ID | Title | Post Type | Author | Created At | URL | Points | Number of Comments | year


**Task 1: Extract the data and build the model**

build a probabilistic model from the training set. It will parse the files in the training set and build a vocabulary with all the words it contains in Title which is At 2018. Then for each word, compute their frequencies and the probabilities of each
class (story, ask_hn, show_hn and poll). Extract the data from 2019 as the testing dataset. To process the texts, fold the Title to lowercase, then tokenize and use the set of resulting word as vocabulary. For each word wi in the training set, save its frequency and its conditional probability for each Post Type class: P(wi|story), P(wi|ask_hn), P(wi|show_hn) and P(wi|poll). These probabilities must be smoothed with 𝛿 = 0.5.

Save model in the text file named **model-2018.txt**. Output file must be sorted alphabetically. The format of this file must be the following:

    1. A line counter i, followed by 2 spaces.
    2. The word wi, followed by 2 spaces.
    3. The frequency of wi in the class story, followed by 2 spaces.
    4. The smoothed conditional probability of wi in story – P(wi|story), followed by 2 spaces.
    5. The frequency of wi in the class ask_hn, followed by 2 spaces.
    6. The smoothed conditional probability of wi in ask_hn – P(wi|ask_hn), followed by 2 spaces. 
    7. The frequency of wi in the class show_hn, followed by 2 spaces.
    8. The smoothed conditional probability of wi in show_hn – P(wi|show_hn), followed by 2 spaces. 
    9. The frequency of wi in the class poll, followed by 2 spaces.
    10. The smoothed conditional probability of wi in poll – P(wi|poll), followed by a carriage return.



**Task 2: Use ML Classifier to test dataset**

use model from task 1 to implement and test a Naive Bays Classifier to classify posts into their likely class in 2019 dataset. To avoid arithmetic underflow, work in log10 space. Run classifier on the 2019 testing dataset and create a single file named **baseline-result.txt** with classification results. For each test file, **baseline-result.txt** should contain:

    1. a line counter, followed by 2 spaces
    2. the name of the test post Title, followed by 2 spaces
    3. the classification as given by your classifier (the label story, ask-hn, show-hn or poll), followed by 2 spaces
    4. the score of the class story as given by your classifier, followed by 2 spaces
    5. the score of the class ask-hn as given by your classifier, followed by 2 spaces
    6. the score of the class show-hn as given by your classifier, followed by 2 spaces
    7. the score of the class poll as given by your classifier, followed by 2 spaces
    8. the correct classification of post, followed by 2 spaces
    9. the label right or wrong (depending on the case), followed by a carriage return.


**Task 3.1: Stop-word Filtering**

Use the baseline experiment and redo tasks 1 and 2 but this time remove the stop words (stopwords.txt) from vocabulary. Generate the new model and result files named **stopword-model.txt** and **stopword-result.txt**.


**Task 3.2: Word Length Filtering**

Use the baseline experiment and redo tasks 1 and 2 but this time remove all words with length ≤ 2 and all words with length ≥ 9. Generate the new model and result files named **wordlength-model.txt** and **wordlength-result.txt**.


**Task 3.3: Infrequent Word Filtering**

Use the baseline experiment, and gradually remove from the vocabulary words with frequency= 1, frequency ≤ 5, frequency ≤ 10, frequency ≤ 15 and frequency ≤ 20. Then gradually remove the top 5% most frequent words, the 10% most frequent words, 15%, 20% and 25% most frequent words. Plot both performance of the classifiers against the number of words left in your vocabulary as two graphs.
