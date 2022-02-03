
import os
import numpy as np
import tensorflow as tf


class TransferLearningImageBaseModel:
    "Base model for image classification."
    
    def __init__(self 
                 , baseModelName = 'MobileNetV2' 
                 , weights = 'imagenet'
                 , imageShape = None
                ):
        
        self.baseModel = None
        self.preProcessInput = None
        if baseModelName == 'MobileNetV2':
            self.baseModel = tf.keras.applications.MobileNetV2(
                input_shape=imageShape,
                include_top=False,
                weights=weights
            )
            self.baseModel.trainable = False
            self.preProcessInput = tf.keras.applications.mobilenet_v2.preprocess_input
            
    def getPreProcessInput(self):
        return self.preProcessInput
    
    def __call__(self, x):
        return self.baseModel(x, training=False)
    
    def get(self):
        return self.baseModel
    