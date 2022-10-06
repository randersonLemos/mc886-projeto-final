import os
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int( 1.0*1024 ))]
    )  


class WrapperMobilenetV3:
    @classmethod
    def set_augmentation_layers(cls, layers):
        cls.augmentation_layers = layers


    def __init__(self, myDataset, base_learning_rate=0.0001):
        self.ds = myDataset
        self.base_learning_rate = base_learning_rate

        self._fine_tune = False


    def load(self):
        data_augmentation = self.augmentation_layers
        image_shape = self.ds.image_shape
        trai = self.ds.trai
        vali = self.ds.vali
        class_names  = self.ds.class_names
        class_num  = len(class_names)

        # MOBILENET_V3 
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        
        # Create the base model from the pre-trained model MobileNet V3
        base = tf.keras.applications.MobileNetV3Small(
              input_shape=image_shape
            , include_top=False
            , weights='imagenet'
            , alpha=0.75
        )

        base.trainable = False

        image_batch, label_batch = next(iter(trai))
        feature_batch = base(image_batch)
        print('feature_batch:        ',feature_batch.shape)

        # Add a classification head
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print('feature_batch_average:',feature_batch_average.shape)

        prediction_layer = tf.keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
        prediction_batch = prediction_layer(feature_batch_average)
        print('prediction_batch:     ', prediction_batch.shape)
        # Build the model
        inputs = tf.keras.Input(shape=image_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        self.base  = base
        self.model = model
        model.summary()
        print('trainable_variables:', len(model.trainable_variables))

        return self


    def train(self, initial_epochs):
        self.initial_epochs = initial_epochs
        model = self.model 
        trai = self.ds.trai
        vali = self.ds.vali
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
        self.fine_tune_epochs = fine_tune_epochs

        trai = self.ds.trai
        vali = self.ds.vali
        base  = self.base
        model = self.model 
        history = self.history   
        initial_epochs = self.initial_epochs
        base_learning_rate = self.base_learning_rate

        # Fine tuning
        base.trainable = True
        print("Number of layers in the base model: ", len(base.layers))

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

        # Compile the model
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            , optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10)
            , metrics=['accuracy']
        )

        model.summary()

        print('trainable variables:',len(model.trainable_variables))

        total_epochs = initial_epochs + fine_tune_epochs
        self.total_epochs = total_epochs

        history = model.fit(
            trai
            , epochs=total_epochs
            , initial_epoch=initial_epochs
            , validation_data=vali
        )

        epoch = self.history.epoch + history.epoch 
        self.history.epoch = epoch

        loss = self.history.history['loss'] + history.history['loss']
        self.history.history['loss'] = loss

        accu = self.history.history['accuracy'] + history.history['accuracy']
        self.history.history['accuracy'] = accu

        loss = self.history.history['val_loss'] + history.history['val_loss']
        self.history.history['val_loss'] = loss

        accu = self.history.history['val_accuracy'] + history.history['val_accuracy']
        self.history.history['val_accuracy'] = accu

        return self


    def results(self):
        history = self.history
        initial_epochs = self.initial_epochs

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
        plt.xlim(0, self.total_epochs-1)
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        if self._fine_tune:
            plt.plot(
                [initial_epochs,initial_epochs]
                , plt.ylim()
                , label='Start Fine Tuning'
            )
 
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.xlim(0, self.total_epochs-1)
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')

        if self._fine_tune:
            plt.plot(
                [initial_epochs,initial_epochs]
                , plt.ylim()
                , label='Start Fine Tuning'
            )

        plt.show()   

        return self
