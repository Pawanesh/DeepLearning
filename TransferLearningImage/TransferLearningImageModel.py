import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

class TransferLearningImageModel:
    "A neural network model based on transfer learning for image classification."
    
    def __init__(self
                 , imageData
                 , baseModel
                 , learningRate = 0.0001
                 , dropouts = 0.2
                 , optimizer = tf.keras.optimizers.Adam()
                 , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                 , metrics = ['accuracy']
                 , modelDirectory = os.environ['PWD']
                 , modelName = 'Model'
                ):
        
        self.imageData = imageData
        self.baseModel = baseModel
        self.learningRate = learningRate
        self.dropouts = dropouts
        self.optimizer = optimizer
        self.optimizer.lr.assign(learningRate)
        self.loss = loss
        self.metrics = metrics
        self.modelDirectory = modelDirectory
        self.modelName = modelName
        
        self.model = self.create()
        self.compile()
        
    def create(self):
        inputs = self.inputLayer()
        x = self.augmentationLayer()(inputs)
        x = self.baseModelLayers(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.dropoutLayer()(x)
        outputs = self.predictionLayer()(x)
        return tf.keras.Model(inputs, outputs)
    
    def inputLayer(self):
        return tf.keras.layers.Input(shape=self.imageData.getImageShape())
    
    def augmentationLayer(self):
        return tf.keras.Sequential([
                    tf.keras.layers.RandomFlip(self.imageData.getRandomFlip()),
                    tf.keras.layers.RandomRotation(self.imageData.getRandomRotation()),
                ])
    
    def baseModelLayers(self, x):
        x = self.baseModel.getPreProcessInput()(x)
        x = self.baseModel(x)
        return x
    
    def dropoutLayer(self):
        return tf.keras.layers.Dropout(self.dropouts)
    
    def predictionLayer(self):
        return tf.keras.layers.Dense(self.imageData.getClassSize())
    

    
    def compile(self):
        self.model.compile(
                    optimizer=self.optimizer
                    , loss=self.loss
                    , metrics = self.metrics
                )
        
    def recompile(self, learningRateFactor):
        self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningRate/learningRateFactor)
                    , loss=self.loss
                    , metrics = self.metrics
                )
    
    def train(self
                , initiaEepochs = 100
                , fineTuneEpochs = 100
                , fineTuneAt = 100
                , learningRateFactor=10,
                verbose = 0):
        self.history = self.model.fit(self.imageData.getTrainDataSet(),
                                epochs=initiaEepochs,
                                validation_data=self.imageData.getValidationDataSet(),
                                verbose = verbose)
        
        
        self.baseModel.get().trainable = True
        
        for layer in self.baseModel.get().layers[:fineTuneAt]:
            layer.trainnable = False
        
        self.recompile(learningRateFactor)
        self.historyFineTune = self.model.fit(self.imageData.getTrainDataSet(),
                                     epochs=initiaEepochs + fineTuneEpochs,
                                     initial_epoch=self.history.epoch[-1],
                                     validation_data=self.imageData.getValidationDataSet())

        
        self.save()
        return self.history, self.historyFineTune
    
    def save(self):
        self.model.save(os.path.join(self.modelDirectory, self.modelName + ".h5"))
        
    def saveLite(self):
        savedDir = os.path.join(self.modelDirectory, 'saved')

        tf.saved_model.save(self.model, savedDir)
        
        converter = tf.lite.TFLiteConverter.from_saved_model(savedDir)
        tfliteModel = converter.convert()
        
        tfliteDir = os.path.join(self.modelDirectory, 'tflite')

        if not os.path.exists(tfliteDir):
            os.makedirs(tfliteDir)

        tfliteModelFile = pathlib.Path(os.path.join(tfliteDir, self.modelName + '.tflite'))
        tfliteModelFile.write_bytes(tfliteModel)

    def load(self, modelPath=None):
        if modelPath is None:
            self.model = tf.keras.models.load_model(os.path.join(self.modelDirectory, self.modelName + ".h5"))
        self.model = tf.keras.models.load_model(modelPath)

    def plotPerformance(self):
        self.plotHistory(self.history, 'History')
        self.plotHistory(self.historyFineTune, 'HistoryFineTune')
        
    def plotHistory(self, history, name):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title(name + ': Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
        
    def summary(self):
        return self.model.summary()

    def predict(self, image_path):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=self.imageData.getImageSize())

        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        label = np.argmax(score)
        return label, self.imageData.getClassName(label)
