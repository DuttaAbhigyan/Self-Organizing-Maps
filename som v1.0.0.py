#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 00:32:47 2018

@author: abhigyan
"""
import numpy as np

"""This is the neighbourhood function class which will contain all the attributes of the
   neighbourhood function to be used by the user. The __init__ method accepts the neighbourhood
   function as function, the parameters (for Gaussian parameters = standard deviation and for
   inverse the formulae followed is (self.parameters[1] /(self.parameters[0] + epochs))).
   The decayRule takes in the function which decides the depreciation/appreciation of parameters."""
   
"""For inverse and exponential appreciation of paramters take place, for Gaussian depreciation takes
   place with the ultimate goal of decreasing Neighbourhood Strength. The decayRateParameters is used
   to paramterize the decayRule function. Standard deviation for gaussian and the negative co-efficient
   for the exponential"""
   

class neighbourhood_functions(object):
    
    def __init__(self, function, parameters, decayRule = None, decayRateParameters = 1):
        self.function = function
        self.parameters = parameters
        self.decayRule = decayRule
        self.decayRateParameters = decayRateParameters
                
        if(self.function == 'inverse' and len(parameters) != 2):
            raise ValueError("Please input proper paramters.")
        elif((self.function == 'step' or self.function == 'exponential' or self.function == 'gaussian') and len(parameters) != 1):
            raise ValueError("Please input proper paramters.")
            
            
    #Will update the neighbourhood function paramters based on epochs. For inverse only one paramter
    #self.parameters[0] is updated
    
    def parameter_updater(self, epochs):
        if(self.decayRule == 'gaussian' and self.function == 'gaussian'):
            self.parameters = self.parameters * np.exp(-np.square(epochs)/(2*self.decayRateParameters))
        
        elif(self.decayRule == 'gaussian' and (self.function == 'exponential' or self.function == 'inverse')):
            self.parameters[0] = self.parameters[0] * np.exp(+np.square(epochs)/(2*self.decayRateParameters))
            
        elif(self.decayRule == 'exponential' and self.function == 'gaussian'):
            self.parameters = self.parameters * np.exp(-epochs * self.decayRateParameters)
            
        elif(self.decayRule == 'exponential' and (self.function == 'exponential' or self.function == 'inverse')):
            self.parameters[0] = self.parameters[0] * np.exp(epochs * self.decayRateParameters)
            
        
    #Calculates NEighbourhood strength        
    def calculate_neighbourhood_coeff(self, distanceMap, epochs):  
        self.parameter_updater(epochs)
        
        if(self.function == 'gaussian'):
            self.neighbourhood_coeff = np.exp(-np.square(distanceMap)/(2*self.parameters[0]))
        elif(self.function == 'exponential'):
            self.neighbourhood_coeff = np.exp(-distanceMap * self.parameters[0])
        elif(self.function == 'inverse'):
            self.neighbourhood_coeff = self.parameters[1] * np.power((distanceMap + self.parameters[0]), -1)
            self.neighbourhood_coeff[len(distanceMap)//2, len(distanceMap[0])//2] = 1
        elif(self.function == 'step'):
            self.neighbourhood_coeff = np.ones(distanceMap.shape) - distanceMap * self.parameters[0]
            
    
    #Getter method        
    def get_neighbourhood_strength(self):
        return self.neighbourhood_coeff
    

#END of neighbourhood_functions       
            

"""Used to create the Self Organizing Map itself, takes in the dimensions of the SOM, Neighbourhood function object of the
   class defined previously, the learning rate and its various other updation and decay rules and the weight initialization
   method"""
   
class Self_Organizing_Maps(object):
    
    def __init__(self, dimensions, NSfunction, distanceFunction = 'Eucledian', learningRate = 0.01, 
                 LRDecayRule = None, LRDecayRate = 0.01, initialization = 'random', learningType = 'batch'):
        self.dimensions = dimensions
        self.distanceFunction = distanceFunction
        self.learningRate = learningRate
        self.LRDecayRule = LRDecayRule
        self.LRDecayRate = LRDecayRate
        self.NSfunction = NSfunction
        self.initialization = 'random'
        self.learningType = learningType
        
        if(len(self.dimensions) == 1):
            self.SOMtype = '1D'
        elif(len(self.dimensions) == 2):
            self.SOMtype = '2D'
            
    
    def create_SOM_structure(self):        
        if(self.SOMtype == '1D'):
            self.create_1D_dMap()
        elif(self.SOMtype == '2D'):
            self.create_2D_dMap()
            
            
    def create_1D_dMap(self):
        leftSide = np.arrange(self.dimensions, 0.0, -1)
        rightSide = np.arrange(1.0, self.dimensions+1, 1)
        
        self.distanceMap = np.concatenate((leftSide, rightSide))
        
        
    def create_2D_dMap(self):
        self.distanceMap = np.zeros((2*self.dimensions[0]-1, 2*self.dimensions[1]-1))
        self.centreCoordinate = np.array([self.dimensions[0]-1, self.dimensions[1]-1])
         
        self.distanceMap[:, :] = np.maximum(np.abs(np.arange(2*self.dimensions[0]-1).reshape(-1, 1) - (self.dimensions[0]-1)),
                                            np.abs(np.arange(2*self.dimensions[1]-1) - (self.dimensions[1]-1)))
        
    
    #Weights for all nodes are initialized here. It creates a new 2D array, even for 2D SOM the weight matrix
    #is 2D and the rule followed to convert it to 2D is row major.    
    def initialize_weights(self, features):
        self.features = features
        
        if(self.initialization == 'random'):
            self.weights = np.random.normal(0, 1, size = (self.features, self.dimensions[0]*self.dimensions[1]))
        elif(self.initilization == 'from_data'):
            self.weights = self.dataset[np.random.choice(self.features, self.dimensions[0]*self.dimensions[1], 
                                                         replace = False), :].T
        
        
    def feature_normalize(self):
        self.dataset = (self.dataset - np.mean(self.dataset, axis = 1, keepdims = True))/np.var(self.dataset, axis = 1, keepdims = True)
    
    #To be called by the user with the dataset and number of epochs
    def run_SOM(self, dataset, epochs, normalize = True):
        self.initialize_weights(dataset.shape[1])
        self.create_SOM_structure()
        self.epochs = epochs
        self.dataset = dataset
        
        for i in range(0, epochs):
            self.distanceMatrix = np.linalg.norm(self.weights[:, :, None] - self.dataset.T[:, None, :], axis = 0)
            self.nodeAssignmentMatrix = np.argmin(self.distanceMatrix, axis = 0).flatten()
            self.changeWeightsMatrix = np.zeros((self.weights.shape))
        
            for i in range(0, len(self.nodeAssignmentMatrix)):
                self.changeWeightsMatrix[:, self.nodeAssignmentMatrix[i]] +=  (self.weights[:, self.nodeAssignmentMatrix[i]]
                                                                             - self.dataset.T[:, i])
            
            self.calculate_weight_updates(i)
            self.adjust_learningRate(i)
            self.update_weights()
            
                
    def calculate_weight_updates(self, epochs):
        self.changeWeightsMatrix = self.changeWeightsMatrix.T.reshape(self.dimensions[0],
                                                                    self.dimensions[1],    #Each n index at first axis corresponds
                                                                    -1)                    #to weight change for n'th node
        self.NSfunction.calculate_neighbourhood_coeff(self.distanceMap, epochs)
        NSmatrix = self.NSfunction.get_neighbourhood_strength()
        self.updateWeightsMatrix = np.zeros(self.changeWeightsMatrix.shape)
        
        for i in range(0, self.dimensions[0]):
            for j in range(0, self.dimensions[1]):
                self.updateWeightsMatrix += ((self.changeWeightsMatrix[self.dimensions[0]-i-1, self.dimensions[1]-j-1, :].reshape(1,1,-1))
                                            *(NSmatrix[i:i+self.dimensions[0], j:j+self.dimensions[1]]))
        
        self.updateWeightsMatrix = self.updateWeightsMatrix.reshape((self.dimensions[0]*self.dimensions[1], 
                                                                     self.features)).T
        
        
    def adjust_learningRate(self, epoch):
        if(self.LRDecayRule == 'gaussian'):
            self.currentLearningRate = self.learningRate * np.exp(-epoch**2 / (2*self.decayRateParameters[0]))
        elif(self.LRDecayRule == 'exponential'):
            self.currentLearningRate = self.learningRate * np.exp(-epoch * self.decayRateParameters[0])
        elif(self.LRDecayRule == 'inverse'):
            self.currentLearningRate = self.learningRate * self.decayRateParameters[1]/(epoch + self.decayRateParameters[0])
        elif(self.LRDecayRule == 'step'):
            self.currentLearningRate = self.learningRate - epoch * self.decayRateParamters[0]
        elif(self.LRDecayRule == None):
            self.currentLearningRate = self.learningRate
            
    
    def update_weights(self):
        self.weights -= self.currentLearningRate * self.updateWeightsMatrix
    
   #Getter methods
    def getWeights(self):
        return self.weights
      
    def get_node_assignment(self):
      return self.nodeAssignmentMatrix
   
#END
