from random import randrange
from csv import reader
import math

# Load a CSV file
def loadDataset(fileName):
	file = open(fileName, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset
# Split a dataset into k folds
def kFoldCrossValidationSplit(dataset, kFolds):
	datasetSplit = list()
	datasetCopy = list(dataset)
	foldSize = int(len(dataset) / kFolds)
	for i in range(kFolds):
		fold = list()
		while len(fold) < foldSize:
			index = randrange(len(datasetCopy))
			fold.append(datasetCopy.pop(index))
		datasetSplit.append(fold)
	return datasetSplit
# Split a dataset based on an attribute and an attribute value
def testSplit(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
# Select the best split point for a dataset
def getSplit(dataset,splitParameter):
   if splitParameter=='entropy':# this is invoked for parameter entropy
    	classValues = list(set(row[-1] for row in dataset))
    	b_index, b_value, b_score, b_groups = 999, 999, 1, None
    	for index in range(len(dataset[0])-1):
    		for row in dataset:
    			groups = testSplit(index, row[index], dataset)
    			ent = entropy(groups, classValues,b_score)
    			if ent < b_score:
    				b_index, b_value, b_score, b_groups = index, row[index], ent, groups
    	return {'index':b_index, 'value':b_value, 'groups':b_groups}
# Create child splits for a node or make terminal
def split(node, maxDepth, minSize, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = toTerminal(left + right)
		return
	# check for max depth
	if depth >= maxDepth:
		node['left'], node['right'] = toTerminal(left), toTerminal(right)
		return
	# process left child
	if len(left) <= minSize:
		node['left'] = toTerminal(left)
	else:
		node['left'] = getSplit(left,splitParameter)
		split(node['left'], maxDepth, minSize, depth+1)
	# process right child
	if len(right) <= minSize:
		node['right'] = toTerminal(right)
	else:
		node['right'] = getSplit(right,splitParameter)
		split(node['right'], maxDepth, minSize, depth+1)
# Calculate accuracy percentage
def accuracyMetric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
# Evaluate an algorithm using a cross validation split
def evaluateID3(dataset, algorithm, kFolds, *args):
	folds = kFoldCrossValidationSplit(dataset, kFolds)
	scores = list()
	for fold in folds:
		trainingSet = list(folds)
		trainingSet.remove(fold)
		trainingSet = sum(trainingSet, [])
		testSet = list()
		for row in fold:
			rowCopy = list(row)
			testSet.append(rowCopy)
			rowCopy[-1] = None
		predicted = algorithm(trainingSet, testSet, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracyMetric(actual, predicted)
		scores.append(accuracy)
	return scores
# Calculate the Entropy for a split dataset
def entropy(groups, classes,b_score):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	ent = 0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			if p > 0 :
				score=(p*math.log(p,2))
		# weight the group score by its relative size i.e Entrpy gain
		ent-=(score*(size/n_instances))
	return ent
# Create a terminal node value
def toTerminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
# Build a decision tree
def buildTree(train, maxDepth, minSize,splitParameter):
	root = getSplit(train,splitParameter)
	split(root, maxDepth, minSize, 1)
	return root
# Print a decision tree
def printTree(node, depth=0):
	if isinstance(node, dict):
		print('%s[ATTRIBUTE[%s] = %.50s]' % ((depth*'\t', (node['index']+1), node['value'])))
		printTree(node['left'], depth+1)
		printTree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
# ID3 Algorithm
def ID3(train, test, maxDepth, minSize,splitParameter):
	tree = buildTree(train, maxDepth, minSize,splitParameter)
	#############
	print('Dictionary Representation of tree on training set\n')
	print(tree)
	print('  ')
	print('Textual Representation of JSON tree\n')
	printTree(tree)
	##############
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

##################################################################################
# loading the traffic accidents record dataset									 #
fileName = 'traffic accidents - after preprocessing.csv'#fileName in csv format	 #
dataset = loadDataset(fileName)													 #
# Tree model creation on training set											 #
kFolds = 10																		 #
maxDepth = 6																	 #	
minSize = 1																		 #
splitParameter='entropy'														 #
#Calculating scores for k fold cross validation by setting kFolds value			 #
print('Implementing k-fold cross validation...')								 #
print('Attributes:-\n')															 #
print(dataset[0])																 #
scores = evaluateID3(dataset, ID3, kFolds, maxDepth, minSize,splitParameter)	 #
print('Scores for 10-fold cross validation: %s' % scores)													 #
print('Mean Accuracy: %.1f%%' % (sum(scores)/float(len(scores))))				 #
##################################################################################



