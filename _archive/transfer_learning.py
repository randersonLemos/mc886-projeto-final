import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf; print('Tensorflow version:',tf.__version__)
devs = tf.config.list_physical_devices(); print('Available devices:', devs)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int( 1.0*1024 ))]
    ) 

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from classes.wrappermobilenetv3 import WrapperMobilenetV3
from classes.dataset import Dataset


def plot_aug(tfDataset, augmentation_layers, figsize=(30, 10), grid=(5,5)):
   plt.figure(figsize=figsize)
   rows, cols = grid
   num = rows*cols

   for images, labels in tfDataset.take(1):
        image = images[0]
        for i in range(num):
            ax = plt.subplot(rows, cols, i + 1)
            augmented_image = augmentation_layers(tf.expand_dims(image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')


# Data augmentation
augmentation_layers = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
        , tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.01)
        , tf.keras.layers.experimental.preprocessing.RandomZoom(
              height_factor=0.1
            , width_factor=0.1
        )
    ]
)


if __name__ == '__main__':
    image_shape = (224, 224, 3)
    batch_size = int( 1*32 )

    ds = Dataset(
         directory='scripts/dataset'
       , image_shape=image_shape
       , batch_size=batch_size
       , validation_split=0.2 
       , seed=123
    )
    #ds.plot(show=True)

    WrapperMobilenetV3.set_augmentation_layers(augmentation_layers)

    wrapper = WrapperMobilenetV3(
          myDataset=ds
        , base_learning_rate=0.001 
    )
    wrapper.load()
    wrapper.train(initial_epochs=10)
    #wrapper.results()

    wrapper.fine_tune(fine_tune_at=25, fine_tune_epochs=10)
    wrapper.results()

    test = ds.test
    if test.cardinality().numpy() == 0:
        test = ds.trai

    loss, accuracy = wrapper.model.evaluate(test)
    print('Test accuracy :', accuracy)

    #Retrieve a batch of images from the test set
    image_batch, label_batch = test.as_numpy_iterator().next()
    predictions = wrapper.model.predict_on_batch(image_batch)
    print('Pred. probabilities:', predictions)

    # Apply a sigmoid since our model returns logits
    predicted_class = np.argmax(predictions, axis=-1)

    print('Predictions:', predicted_class)
    print('Labels:     ', label_batch)

    path = pathlib.Path('models')
    path.mkdir(exist_ok=True)

    try:
        nxt = int ( sorted( [ el for el in map( str, path.iterdir() ) if 'mobileV3' in el] )[-1].split('_')[-1] ) + 1
    except:
        nxt = 1

    tf.saved_model.save(wrapper.model, 'models/mobileV3_{}'.format(nxt))
