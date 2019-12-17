import matplotlib.pyplot as plt
import math
import operator
import random
from random import randrange
import numpy as np 

ETA = 0.2
attribute_number=4
instance_number=90
class_label_number=2

def loadDataset(dataset, data=[]):
    
    newdata = []
    for x in range(len(dataset)):
        for i in range(0,len(dataset[x]),4):
            if dataset[x][i] == "I":
                newdata.append(dataset[x][i:len(dataset[x])-1])
                break
            else:
                attribute = float(dataset[x][i:i+3])
                newdata.append(attribute)
            
        data.append(newdata)
        newdata=[]         


class perceptron_learning():
    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0
 
    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self, train, weights, l_rate, n_epoch):
        #weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = self.predict(row, weights)
                error = row[-1] - prediction
                sum_error += error**2
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            
            #print("wei",weights)
        return weights

class K_NN():
    
    def __init__(self):
        self.training_set = []
        self.class_labels = []  
    
    def add_to_training_set(self, example, class_label):
        found = False
        for i in range(len(self.class_labels)):
            if self.class_labels[i] == class_label:
                found = True
                break
        
        if not found:
            self.class_labels.append(class_label)
        self.training_set.append(example)
            
    def calculate_error(self, example1, example2):
        summ = 0.0
        for i in range(len(example1)-1):
            summ += pow((example1[i] - example2[i]), 2)
        summ = math.sqrt(summ)
        
        return summ
    
    def classifier(self, example):
        index = 0
        error = 1000000
        
        for s in self.training_set:
            new_error = self.calculate_error(s, example)
            if error >= new_error:
                index = s[-1]
                error = new_error
        return index
 
    def getNeighbors(self, trainingSet, testInstance):
        distances = []
        length = len(testInstance)-1
        print(length)
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(3): # 3NN
            neighbors.append(distances[x][0])
        return neighbors
 
    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1 
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]
    

class adaboost():

    def __init__(self):
        self.training_set = []
        self.test = []
        self.class_labels = []
        self.px = []
        self.classifier_textbook = [] 
        self.classifier_original = []
        self.Induce_Classifier_Textbook = [K_NN() for i in range(9)]# training_subset_number
        self.Induce_Classifier_Original = [K_NN() for i in range(9)]

    def add_to_training_set(self, example, class_label):
        found = False
        for i in range(len(self.class_labels)):
            if self.class_labels[i] == class_label:
                found = True
                example.append(i)
                break
        
        if not found:
            example.append(len(self.class_labels))
            self.class_labels.append(class_label)
        self.training_set.append(example)
        
    def add_to_testset(self, example, class_label):
        found = False
        for i in range(len(self.class_labels)):
            if self.class_labels[i] == class_label:
                found = True
                example.append(i)
                break
        
        if not found:
            example.append(len(self.class_labels))
            self.class_labels.append(class_label)
        self.test.append(example)
        
    def get_example(self):
        d = random.random()
        for i in range(len(self.px)):
            if d < self.px[i]:
                return self.training_set[i]
            d -= self.px[i]
            
        return self.training_set[-1]
    
    def create_subset(self):
        subset = []
        for i in range(10): # element number in subset = 10
            ex = self.get_example()
            subset.append(ex)
        return subset
    
    def subset_eval_textbook(self, Round):
        epsilon = 0
        BETA = 0
        check = []
        
        for i in range(len(self.training_set)):
            ex = self.training_set[i]
            label = self.Induce_Classifier_Textbook[Round].classifier(ex)
            if label != ex[attribute_number]: # misclassify
                epsilon += self.px[i]
                #print("example:",i,"misclassify")
            
            check.append(label == ex[-1])

        
        print("error",epsilon)
        BETA = float(float(epsilon) / float(1-epsilon))
        
        for i in range(len(self.training_set)):
            if check[i] == True:
                self.px[i] *= BETA

        summ = sum(self.px) # sum up p*error
        self.px = [i / summ for i in self.px] # normalized
            
    def subset_eval_original(self, Round):
        epsilon = 0
        BETA = 0
        check = []
        
        for i in range(len(self.training_set)):
            ex = self.training_set[i]
            label = self.Induce_Classifier_Original[Round].classifier(ex)
            if label != ex[attribute_number]:
                epsilon += self.px[i]
                #print("example:",i,"misclassify")
            
            check.append(label == ex[-1])
        
        print("error",epsilon)
        # different from textbook version
        BETA = pow(float(float(epsilon) / float(1-epsilon)), 0.5)
        
        for i in range(len(self.training_set)):
            if check[i]:
                self.px[i] *= BETA
            else:
                self.px[i] /= BETA

        summ = sum(self.px)
        self.px = [i / summ for i in self.px]
        
        return epsilon
            
    def subset_classification_textbook(self, Round):
        #print("induce classifier:", Round, "subset classification(text)")
        subset = self.create_subset()
        for s in subset:
            self.Induce_Classifier_Textbook[Round].add_to_training_set(s, self.class_labels[s[-1]])
            
        self.subset_eval_textbook(Round)
        
    def subset_classification_original(self, Round):
        #print("induce classifier:", Round, "subset classification(origin)")
        subset = self.create_subset()
        for s in subset:
            self.Induce_Classifier_Original[Round].add_to_training_set(s, self.class_labels[s[-1]])
            
        epsilon = self.subset_eval_original(Round)
        return epsilon
        
    def textbook_version_classify(self):
        self.px = [float(1/len(self.training_set)) for i in range(len(self.training_set))]
        epsilon = []
        print("==========================textbook version==============================")
        for i in range(9):
            epsilon.append(self.subset_classification_textbook(i)) # updating p
        
        classifier_weights = self.master_classifier_textbook(epsilon)
        predict = self.master_classifier_textbook_predict(classifier_weights)

        print("predict accuracy:",(predict/float(len(self.test))) * 100.0)          
            
    def original_version_classify(self):
        epsilon = 0
        self.px = [float(1/len(self.training_set)) for i in range(len(self.training_set))]
        print("==========================original version============================")
        for i in range(9):
            epsilon = self.subset_classification_original(i)            
        
        classifier_weights = self.master_classifier_original(epsilon)
        predict = self.master_classifier_original_predict(classifier_weights)

        print("predict accuracy:",(predict/float(len(self.test))) * 100.0)       
    
    def master_classifier_textbook(self, epsilon): # use perceptron update weights
        classifier_textbook_W = [0.2 for i in range(9)]
        # update classifiers weights
        subset = self.create_subset()
        classifier_textbook_W = perceptron_learning().train_weights(self.training_set, classifier_textbook_W, 0.2, 5)
        print("textbook classifier weights", classifier_textbook_W)
        correct = 0
        for i in range(len(self.training_set)):
            pos, neg = 0.0, 0.0
            example = self.training_set[i]
            for j in range(9):
                subset_classifier_label = self.Induce_Classifier_Textbook[j].classifier(example)
                if(subset_classifier_label):
                    pos += classifier_textbook_W[j]
                else:
                    neg += classifier_textbook_W[j]
            
            master_label = 0
            master_label = 1 if pos > neg else 0
            
            if(example[-1] == master_label):
                correct +=1

        print("textbook training accuracy:", (correct/float(len(self.training_set))) * 100.0)
        return classifier_textbook_W
    
    def master_classifier_textbook_predict(self, classifier_W):
        correct = 0
        for i in range(len(self.test)):
            pos, neg = 0.0, 0.0
            example = self.test[i]
            for j in range(9):
                subset_classifier_label = self.Induce_Classifier_Textbook[j].classifier(example)
                if(subset_classifier_label):
                    pos += classifier_W[j]
                else:
                    neg += classifier_W[j]
            
            master_label = 0
            master_label = 1 if pos > neg else 0
            
            if(example[-1] == master_label):
                correct += 1

        return correct
    
    def master_classifier_original(self, epsilon):
        # create wieghts without updating
        classifier_original_W = [0.5*np.log((1 - epsilon) / epsilon) for i in range(9)]
        print("original classifier weights", classifier_original_W)
        correct = 0
        for i in range(len(self.training_set)):
            pos, neg = 0.0, 0.0
            example = self.training_set[i]
            for j in range(9):
                subset_classifier_label = self.Induce_Classifier_Original[j].classifier(example)
                if(subset_classifier_label):
                    pos += classifier_original_W[j]
                else:
                    neg += classifier_original_W[j]
            
            master_label = 0
            master_label = 1 if pos > neg else 0 # voting
            
            if(example[-1] == master_label):
                correct += 1          
                
        print("original training accuracy:", (correct/float(len(self.training_set))) * 100.0)
        return classifier_original_W
                    
    def master_classifier_original_predict(self, classifier_W):
        correct = 0
        for i in range(len(self.test)):
            pos, neg = 0.0, 0.0
            example = self.test[i]
            for j in range(9):
                subset_classifier_label = self.Induce_Classifier_Original[j].classifier(example)
                if(subset_classifier_label):
                    pos += classifier_W[j]
                else:
                    neg += classifier_W[j]
            
            master_label = 0
            master_label = 1 if pos > neg else 0
            
            if(example[-1] == master_label):
                correct += 1

        return correct
                    

class origin_perceptron():
    
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.weights = [0.2 for i in range(len(train[0]))]
        
    def process_label(self, data):
        for row in data:
            if row[-1] == "Iris-setosa":
                row.pop(-1)
                row.append(0)
            else:
                row.pop(-1)
                row.append(1)
        #print(data)
        return data

    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0
    
    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, n_folds, *args):
        self.train = self.process_label(self.train)
        folds = self.cross_validation_split(self.train, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted= self.perceptron(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)

        return scores

    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    def train_weights(self, train, learning_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = self.predict(row, self.weights)
                #print(prediction)
                error = row[-1] - prediction #list indices must be integers or slices, not list
                sum_error += error**2
                self.weights[0] = self.weights[0] + learning_rate * error
                for i in range(len(row)-1):
                    self.weights[i+1] = self.weights[i+1] + learning_rate * error * row[i]
            #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
            
        print(self.weights)
        return self.weights 


    def perceptron(self, train, test, l_rate, n_epoch):
        predictions = list()
        self.weights = self.train_weights(train, l_rate, n_epoch)
        for row in test:
            prediction = self.predict(row, self.weights)
            predictions.append(prediction)
            
        return predictions
    
    def predict_test(self, test, l_rate, n_epoch):
        test = self.process_label(test)
        predictions = list()
        scores = list()
        print("w",self.weights)
        for row in test:
            prediction = self.predict(row, self.weights)
            predictions.append(prediction)

        actual = [row[-1] for row in test]
        
        print("s",predictions)
        accuracy = self.accuracy_metric(actual, predictions)
        scores.append(accuracy)
        
        print('perceptron predict accuracy:', (sum(scores)/float(len(scores))))
        
def main():
    
    trainingSet=[]
    testSet=[]
    
    f = open('training-data.txt', "r")
    train = f.readlines()
    train = list(train)
    f = open('testing-data.txt', "r")
    test = f.readlines()
    test = list(test)

    loadDataset(train, trainingSet)
    loadDataset(test, testSet)
    
    classifier = adaboost()
    Linear_classifier = origin_perceptron(trainingSet, testSet)
    
    for i in range(90):
        classifier.add_to_training_set(trainingSet[i][:4], trainingSet[i][-1])
        
    for i in range(10):
        classifier.add_to_testset(testSet[i][:4], testSet[i][-1])
        
    classifier.original_version_classify()
    classifier.textbook_version_classify()
    print("==========================perceptron version============================")
    scores = Linear_classifier.evaluate_algorithm(trainingSet, 9, 0.2, 50)
    print('Scores: %s' % scores)
    print('perceptron training accuracy:', (sum(scores)/float(len(scores))))
    Linear_classifier.predict_test(testSet, 0.2, 50)


main()