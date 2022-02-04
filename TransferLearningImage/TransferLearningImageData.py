import os
import numpy as np
import tensorflow as tf

class TransferLearningImageData:
    "Image data for transfer learning model data."
    
    def __init__(self
                , imageSize = (160,160)
                , batchSize = 32
                , shuffle = True
                , trainPath = None
                , testPath = None
                , validationPath = None
                , randomFlip = 'horizontal'
                , randomRotation = 0.2):
        
        self.imageSize = imageSize
        self.imageShape = imageSize + (3,)
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.trainPath = trainPath
        self.testPath = testPath
        self.validationPath = validationPath
        self.trainData = None
        self.testData = None
        self.validationData = None
        self.randomFlip = randomFlip
        self.randomRotation = randomRotation
        
        self.load()
        
    def load(self):
        if self.trainPath:
            self.trainData = tf.keras.utils.image_dataset_from_directory(
                                                    self.trainPath,
                                                    shuffle=self.shuffle,
                                                    batch_size = self.batchSize,
                                                    image_size=self.imageSize
                                                )
            
            
        if self.testPath:
            self.testData = tf.keras.utils.image_dataset_from_directory(
                                                    self.testPath,
                                                    shuffle=self.shuffle,
                                                    batch_size = self.batchSize,
                                                    image_size=self.imageSize
                                                )
            
        if self.validationPath:
            self.validationData = tf.keras.utils.image_dataset_from_directory(
                                                    self.validationPath,
                                                    shuffle = self.shuffle,
                                                    batch_size = self.batchSize,
                                                    image_size = self.imageSize
                                                )
            
    def getTrainDataSet(self):
        return self.trainData
    
    def getTestDataSet(self):
        return self.testData
    
    def getValidationDataSet(self):
        return self.validationData
    
    def getImageSize(self):
        return self.imageSize
    
    def getImageShape(self):
        return self.imageShape
    
    def getClassSize(self):
        return len(self.trainData.class_names)
    
    def getRandomFlip(self):
        return self.randomFlip
    
    def getRandomRotation(self):
        return self.randomRotation

    def getClassName(self, label):
        return self.trainData.class_names[label]
        