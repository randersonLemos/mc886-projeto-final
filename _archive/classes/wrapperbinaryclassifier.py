import os
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class WrapperBinaryClassifier:
    @classmethod
    def set_augmentation_layers(cls, layers):
        cls.augmentation_layers = layers


    def __init__(self, myDataset, base_learning_rate=0.0001):
        self.ds = myDataset
        self.base_learning_rate = base_learning_rate

        self._fine_tune = False


    def load(self):
        data_augmentation = self.augmentation_layers
        image_shape   = self.ds.image_shape
        class_names  = self.ds.class_names
        class_num  = len(class_names)

        base = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(class_num)
        ])

        # Build the model
        inputs = tf.keras.Input(shape=image_shape)
        x = data_augmentation(inputs)
        outputs = base(x)
        self.model = tf.keras.Model(inputs, outputs)

        return self


    def train(self, initial_epochs):
        self.initial_epochs = initial_epochs
        trai = self.ds.trai
        vali = self.ds.vali
        model = self.model 
        base_learning_rate = self.base_learning_rate

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
            , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            , metrics=['accuracy']
        )
        # Train the model
        loss0, accuracy0 = model.evaluate(vali)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        history = model.fit(
            trai
            , epochs=initial_epochs
            , validation_data=vali
        )

        self.history = history

        return self


    def fine_tune(self, fine_tune_at, fine_tune_epochs):
        self._fine_tune = True

        base_model = self.base
        model = self.model
        base_learning_rate = self.base_learning_rate
        history = self.history   
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset
        initial_epochs = self.initial_epochs

        # Fine tuning
        base_model.trainable = True
        print("Number of layers in the base model: ", len(base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = fine_tune_at

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Compile the model
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10),
                      metrics=['accuracy'])

        model.summary()

        print('trainable variables:',len(model.trainable_variables))


        fine_tune_epochs = fine_tune_epochs
        total_epochs =  initial_epochs + fine_tune_epochs

        history_fine = model.fit(train_dataset,
                                 epochs=total_epochs,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=validation_dataset)

        self.history_fine = history_fine
        return self


    def results(self):
        if self._fine_tune:
            self._results_fine()
            return
        self._results()


    def _results(self):
        history = self.history

        # Show the learning curves of the training and validation accuracy/loss
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
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()   

        return self


    def _results_fine(self):
        history = self.history
        initial_epochs = self.initial_epochs

        # Show the learning curves of the training and validation accuracy/loss
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        history_fine = self.history_fine

        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([initial_epochs-1,initial_epochs-1],
                  plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs-1,initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
