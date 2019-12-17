# ML_Programming_Assignment_5
## Adaboost implementation 
##### Using python
#### There are some functions:
### 1. Handle Data
* LoadDataSet
    1. load training-data.txt file 
        * Consume the data into a more convenient file for later implementation
        
### 2. class perceptron_learning():
#### (This class is for updating weights of textbook classifiers)
* predict
    1. Set the bias (W0) as 0.2
        * Multiply each weight with  attribute one by one and sum them up
        * If result >= 0 the label will set as 1, else as 0

* train_weights
    1. Set initial weights to 0.2
    2. Predict the result
    3. Use fomula ![](https://i.imgur.com/mCcyuT7.jpg) to update the weights in each epoch
    4. If the error rate (correct_label - predict_label) gets 0, calculate the example-presentation
    5. return final weights and example-presentation
    
### 3. class K_NN():
* add_to_training_set
    1. Change to label of dataset into 0 and 1
    
* classifier
    1. Updating error of 3 NN classifier prediction
    2. Return the predicted label
    
* calculate_error
    1. Calculate the Euclidean Distance
        * Get the total number of numeric factors -> length
        * Use formula pow((a-b),2) and sum them up

* getNeighbors
    1. Users input k
    2. Using already exists euclideanDistance function to get the similarity from training instances and testing instances
        * All training_examples to use for one test_example
        * `distance.append` pairs of `(training_example[x], distance)`
        * `sort(key=operator.itemgetter(1))` will give: 
        A *function* that grabs the *index=1* item from a list-like object
        
    3. Get K closest neighbors and return then

* getResponse
    1.  Allowing each neighbor to vote for their class attribute, and take the majority vote as the prediction.
        * Assumes that the class is the last attribite for each neighbor
         
### 4. class adaboost(): 
     
* add_to_training_set & add_to_testset
    * Handle data in to training_set list and class_labels list
* get_example
    * Randomly decide if the example be included in the sebset(according to last possibilities)
* create_subset 
    * create subset (10 examples for a subset)
* subset_eval_textbook
    1. calculate epsilon by possibility and predicted result by sub_classifiers (use 3 NN)
        * If 3 NN prediction is the same as true label, ex = 1, else ex = 0.
        * Multiply and sum up the `ex*p` result to calculate epsilon. `𝜖i = S j=1 m   pi(xj)ei(xj) `
        * Calculate Beta by epsilon. `𝛽i =𝜖i  ∕ (1-𝜖i) `
    2. Modify the probabilities of correctly classfied examples.
        * By  `pi+1(xj) = pi(xj) · 𝛽i `
    3. Normalize the probabilities to make sure that the summation of probabilities is 1
    4. Continue to create next classifier
    
* subset_eval_original
    1. Quite similar as textbook version, the only differences are 
        * ![](https://i.imgur.com/Mey2Kxg.jpg)

        * `𝛽1 =(𝜖1 ∕(1-𝜖1))^1/2 `
        * probability update by 
            * correct examples: `p2(xj) = p1(xj) ·𝛽1`

            * incorrect examples : `p2(xj) = p1(xj) /𝛽1
    
* subset_classification_textbook
    1. Use classifiers to classify each subset(by 3 NN)
* subset_classification_original
    1. Use classifiers to classify each subset(by 3 NN)
    2. return epsilon for master classifier to use
* textbook_version_classify
    1. Initialize probabilities by 1/ 90
    2. Use `subset_classification_textbook` for 9 times to classify 9 subsets
    3. Use `master_classifier_textbook` to modify classifiers' weights
    4. Use `master_classifier_textbook_predict` to predict testing-data.txt
    5. Get the accuracy of prediction
    
* original_version_classify
    1. Initialize probabilities by 1/ 90
    2. Use `subset_classification_original` for 9 times to classify 9 subsets
    3. Use `master_classifier_original` to modify classifiers' weights
    4. Use `master_classifier_original_predict` to predict testing-data.txt
    5. Get the accuracy of prediction
   
* master_classifier_textbook
    1. Create classifiers' weights (init 0.2)
    2. Use 3NN to classify all trainingset (90 examples) by 9 classifiers
    3. Modify classifiers' weights by perceptron learning algorithm(If the accuracy is high, then the weight will be heavy)
    4. Use the weights to do the master voting
        * If correct: pos
        * Ir incorrect: neg
        * Compare pos and neg to vote the final prediction
    5. Compare the prediction with correct answer to get the accuracy
    6. Show accuracy
* master_classifier_original
    1. Create classifiers' weights 
        * init by ![](https://i.imgur.com/jXb5Awo.jpg)

    2. Use the weights to do the master voting
        * No need to update weights
        * If correct: pos
        * Ir incorrect: neg
        * Compare pos and neg to vote the final prediction
    3. Compare the prediction with correct answer to get the accuracy
    4. Show accuracy

* master_classifier_textbook_predict
    1. Similar as `master_classifier_textbook`, only get the finished-updating classifier weights to predict testing-data.txt

* master_classifier_original_predict 
    1. Similar as `master_classifier_original`, only get the original-initialed weights(`at`) to predict testing-data.txt
### 5. class original_perceptron():
* process_label
    * Preocess the data and change labels to 0/1
* predict
    * Same as described above
* accuracy_metric
    * Calculate accuracy in percentage 
* evaluate_algorithm
    1. Evaluate the algorithm using a cross validation split.
    2. Use `cross_validation_split` to split the training-set to 9 folds.
    3. Use each folds to train the model by `perceptron` and modify each weights.
* cross_validation_split
    1. Since if we give the model all training-set data, the accuracy of training result will be bad, so I split the data into many folds to do the training process.
* train_weights
    * Same as described above
* perceptron
    1. Use `train_weights` to get the modified weights.
    2. Use `predict` to predict the testcase
        * in this case, training set, actually 
* predict_test
    * Predict the testing-set and get the accuracy.
### 6. Main()

1. Call each classes to get the training and prediction results.
###### Adaboost original version:
![](https://i.imgur.com/gFxtIG5.jpg)

###### Adaboost textbook version:
![](https://i.imgur.com/6B5JOKL.jpg)

###### Percpetron learning version:
![](https://i.imgur.com/TTmvcYh.jpg)

2. Observation:
    * After done many times of training, I found out that the average accuracy is original_version_adaboost > textbook_version_adaboost > perceptron_learning
    * It's not surprise cause adaboost is a method to improve original classifier(in this practice, perceptron algorithm)
