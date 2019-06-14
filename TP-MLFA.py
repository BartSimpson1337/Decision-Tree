import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Global variables, to calculate the tree size
max_depth = 0
depths = []
total_nodes = 0
total_leafs = 0
total_rows = 0

pruning = False
pruning_depth = False
pruning_error = False
pruning_data = False

#Defined Functions
#Reding CSV Wine Dataset
def ReadCSV():
    return np.array(pd.read_csv('./pima-indians-diabetes.csv',sep=';').values[:,:],dtype=np.float).round(decimals=2).tolist()

def TrainTestSplit (data):
    
    X = np.asarray(data)[:, :-1] #Getting all columns from dataset except the last one.
    Y = np.asarray(data)[:, -1] #Getting class feature
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=0) #Train test 80/20
  
    trainingData = np.column_stack((xTrain,yTrain)) #Appending y train to x train
    testingData = np.column_stack((xTest,yTest)) #Appending y test to x test 
    return trainingData.tolist(), testingData.tolist()


def K_Fold_CrossValidation(X, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation

def CrossValidationScorer(rows):

    scores = []
    X = rows
    for training, validation in K_Fold_CrossValidation(X, K=10):
        model = BuildTree(training, 0) #construi com 1 nivel e depois 2 niveis ...
        classifications = Scorer(validation, model)
        accuracy, precision, recall, matrix = ConfusionMatrix(classifications)
        scores.append(accuracy)
    score = np.mean(scores)

    return score

def BuildTree(rows, depth):

    if pruning == True:

        #Recursive variable, that increments on every call to function BuildTree, passing it has a parameter.
        depth = depth + 1 

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, attribute = FindSplit(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0 or (depth >= max_depth and pruning_depth == True):
            depths.append(depth)
            return Leaf(rows)

        
        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = Partition(rows, attribute)

        classification_error, classification_error_left, classification_error_right = ClassificationError_Pruning(true_rows, false_rows)
        if ((classification_error_left >= classification_error or classification_error_right >= classification_error) and (depth > 5)) and pruning_error == True: 
            depths.append(depth)
            return Leaf(rows)
        else:

            if MinimumDataPoints_Pruning(len(true_rows)+len(false_rows)) == False and pruning_data == True:
                depths.append(depth)
                return Leaf(rows)
            else:
                # Recursively build the true branch.
                true_branch = BuildTree(true_rows, depth)

                # Recursively build the false branch.
                false_branch = BuildTree(false_rows, depth)

                # Return a Question node.
                # This records the best feature / value to ask at this point,
                # as well as the branches to follow
                # dependingo on the answer.
                return Node(attribute, true_branch, false_branch)
    else:

        #Recursive variable, that increments on every call to function BuildTree, passing it has a parameter.
        depth = depth + 1 

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, attribute = FindSplit(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0:
            depths.append(depth)
            return Leaf(rows)

        
        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = Partition(rows, attribute)

        # Recursively build the true branch.
        true_branch = BuildTree(true_rows, depth)

        # Recursively build the false branch.
        false_branch = BuildTree(false_rows, depth)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # dependingo on the answer.
        return Node(attribute, true_branch, false_branch)


def FindSplit(rows):
    best_gain = 0
    best_attribute = None
    current_uncertainty = Gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique values in the column
        
        for val in values:  # for each value
            attribute = Attribute(col, val)
             # try splitting the dataset
            true_rows, false_rows = Partition(rows, attribute)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0: #Verificar possível prunning.
                continue

            # Calculate the information gain from this split
            gain = InformationGain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_attribute = gain, attribute

    return best_gain, best_attribute

#Impurity calculation using Gini
def Gini(rows):
    counts = ClassCounting(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

#Information Gain based on Gini impurity 
def InformationGain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * Gini(left) - (1 - p) * Gini(right)



def ClassCounting(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
    
        label = row[-1] #Getting the last column, which represents the label.
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def Partition(rows, attribute):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if attribute.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def PrintTree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.attribute))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    PrintTree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    PrintTree(node.false_branch, spacing + "  ")


def Scorer(testing_data, model):

    classifications = []
    for index, row in enumerate(testing_data, start = 0):
        classifications.append([])
        classifications[index].append(row[-1])
        classifications[index].append(list(Classify(row, model).keys())[0])
        
    return classifications

def Classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.attribute.match(row):
        return Classify(row, node.true_branch)
    else:
        return Classify(row, node.false_branch)


def ConfusionMatrix(classifications):
    TP = 0
    TN = 0 
    FN = 0
    FP = 0

    for classification in classifications:
        #Counting the number of true positives and true negatives
        if classification[0] == classification[1]:
            if classification[0] == 1 and classification[1] == 1:
                TP += 1
            else:
                TN += 1
        else:
            #Counting the number of false positives and false negatives 
            if classification[0] == 1 and classification[1] == 0:
                FN += 1
            else:
                FP += 1

    try:

        matrix = np.matrix([[TP, FP], [FN, TN]])
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except Exception:
        matrix = 0
        recall = 0
        precision = 0
        accuracy = 0

    return accuracy, precision, recall, matrix 


"""
    Pruning functions, to optimize the decision tree model.
"""

#Function that will determine, the best depth to build a decision tree model.
def OptimalDepth_Pruning(rows):

    best_depth = 0
    #Executing this function, only when pruning has been set to true.
    if pruning_depth == True:

        best_score = 0
        tmp_score = 0 

        global max_depth
        max_depth = 5 #Starting to build trees with an meximum of depth = 5 
        while True: #Loop, that will search for the best score, given by cross validation
            score = CrossValidationScorer(rows) 
    
            if (score+0.001 > best_score and score != tmp_score): #Giving priority to scores with values, and acceptable depth.
                best_score = score
                best_depth = max_depth   
            elif score == tmp_score: #stopping the loop, when the accuracy between interactions, has not changing at all
                break
            else:
                tmp_score = score #Saving the last interaction

            max_depth = max_depth + 1 #Incrementing tree size threshold.
        
        #Cleaning variables, before leaving this function 
        global total_nodes 
        total_nodes = 0
        global total_leafs
        total_leafs = 0   
        global depths 
        depths = []

    return best_depth


def ClassificationError_Pruning(true_rows, false_rows):
    classification_error = len(true_rows)/(len(true_rows) + len(false_rows))
    classification_error_left = 0
    classification_error_right = 0
    
    #Validate each branch 
    if len(true_rows) > 0:
        gain, attribute = FindSplit(true_rows)

        if gain != 0:
            true_rows_left, false_rows_left = Partition(true_rows, attribute)
            classification_error_left = len(true_rows_left)/(len(true_rows_left) + len(false_rows_left))

    if len(false_rows) > 0:
        gain, attribute = FindSplit(false_rows)

        if gain != 0:
            true_rows_right, false_rows_right = Partition(false_rows, attribute)
            classification_error_right = len(true_rows_right)/(len(true_rows_right) + len(false_rows_right))

    return classification_error, classification_error_left, classification_error_right


def MinimumDataPoints_Pruning(total_rows_node):
    global total_rows
    partition_size = (total_rows_node * 100) / total_rows

    if partition_size <= 5:
        return False
    return True


#Defined Classes
class Attribute:
    """A Attribute is used to partition a dataset.
    This class just records a 'column number' and a
    'column value' . The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. 
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def isNumeric(self, value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if Attribute.isNumeric(self, val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        #Dataset column headers
        header = ["Number of times pregnant","glucose tolerance test","Diastolic blood pressure","Triceps skin fold thickness","2-Hour serum insulin","Body mass index","Diabetes pedigree","Age","Class"]

        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if Attribute.isNumeric(self, self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = ClassCounting(rows)
        global total_leafs
        total_leafs = total_leafs + 1 #Incrementing variable.

    

class Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 attribute,
                 true_branch,
                 false_branch):
        self.attribute = attribute
        self.true_branch = true_branch
        self.false_branch = false_branch
        
        global total_nodes
        total_nodes = total_nodes + 1 #Incrementing variable.


"""Area that will be used to call functions and display output results. 
"""
#Setting pruning paremeters
pruning = True
pruning_depth = False
pruning_error = False
pruning_data = True


#Splitting the dataset in train and data.
training_data, testing_data = TrainTestSplit(ReadCSV())
total_rows = len(training_data) 

max_depth = OptimalDepth_Pruning(training_data) #This function will retrieve data if parameter has been set to True

#Builind tree model with training data.
model = BuildTree(training_data, 0)

#Printing tree model
PrintTree(model)


#Validating the model with testing data, and displaying the classifications.
classifications = Scorer(testing_data, model)
print('Total instances:', len(testing_data))
for classification in classifications:
    print ("Actual: %s. Predicted: %s" % (classification[0], classification[1]))


#Displaying information about the tree size.
print('Tree Size:', total_nodes, ' Nodes, ', total_leafs, ' Leafs, ', np.amax(depths), 'Depth', '\n')
#Calculating the confusion matrix, and displaying it.
accuracy, precision, recall, matrix = ConfusionMatrix(classifications)
print("\n")
print('Confusion Matrix')
print(matrix)
print('Accuracy: %.2f%% Recall: %.2f%% Precision: %.2f%%' % (accuracy, recall, precision))
print("\n")

