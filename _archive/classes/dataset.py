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

class Dataset:
    def __init__(self, directory, image_shape, batch_size, validation_split, seed=123):
        self.directory   = directory
        self.image_size  = image_shape[:2]
        self.image_shape = image_shape
        self.batch_size  = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.trai = None
        self.vali = None
        self.test = None
        self.class_names = ''

        self._load()


    def _load(self):
        self.trai = tf.keras.preprocessing.image_dataset_from_directory(
              directory=self.directory
            , image_size=self.image_size
            , batch_size=self.batch_size
            , validation_split=self.validation_split
            , subset="training"
            , seed=self.seed
        )

        self.vali = tf.keras.preprocessing.image_dataset_from_directory(
              directory=self.directory
            , image_size=self.image_size
            , batch_size=self.batch_size
            , validation_split=self.validation_split
            , subset="validation"
            , seed=self.seed
        )

        cardinality = tf.data.experimental.cardinality(self.vali)
        self.test = self.vali.take(cardinality // 5)

        self.class_names = self.trai.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        self.trai = self.trai.prefetch(buffer_size=AUTOTUNE)
        self.vali = self.vali.prefetch(buffer_size=AUTOTUNE)
        self.test = self.test.prefetch(buffer_size=AUTOTUNE)

        print('Train dataset:      {} bacthes of size {}'.format(tf.data.experimental.cardinality(self.trai), self.batch_size))
        print('Validation dataset: {} bacthes of size {}'.format(tf.data.experimental.cardinality(self.vali), self.batch_size))
        print('Test dataset:       {} bacthes of size {}'.format(tf.data.experimental.cardinality(self.test), self.batch_size))
        print('Classes name:       {}'.format(self.class_names))

        return self


    def update_train(self, tfDataset):
        try:
            ds = tfDataset.unbatch()
        except ValueError:
            ds = tfDataset

        self.trai = ds.batch(batch_size=self.batch_size)


    def plot(self, figsize=(30, 10), grid=(5,5), show=False):
        plt.figure(figsize=figsize)
        rows, cols = grid
        num = rows*cols
        for i, ( image, label ) in zip(range(num), self.trai.unbatch()):    
            ax = plt.subplot(rows, cols, i + 1)
            plt.imshow(image.numpy().astype("uint8"))
            plt.title(self.class_names[label], y=-0.20)
            plt.axis("off")
        plt.tight_layout()


        if show:
            plt.show()

    def show(self):
        plt.show()


class Viewer:
    @classmethod
    def viewplt(cls, tfDataset, num=25):
        try:
            ds = tfDataset.unbatch()
        except ValueError:
            ds = tfDataset


        plt.figure(figsize=(30, 10))
        max_num = min(25, num)
        for en, ( image, label ) in enumerate(ds):
            ax = plt.subplot(5, 5, en + 1)
            mat = image.numpy().astype('uint8')
            plt.imshow(mat)
            plt.title(str(label.numpy()), y=-0.20)
            plt.axis("off")
            if en+1 == max_num:
                break


    @classmethod
    def view(cls, tfDataset, num=-1, waitKey=1, title='Dataset'):
        try:
            ds = tfDataset.unbatch()
        except ValueError:
            ds = tfDataset

        for en, ( image, label ) in enumerate( ds ):
            mat = cv2.cvtColor(image.numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
            cv2.imshow( title, mat )
            cv2.waitKey(waitKey)


def func_augment001(image_label, seed):
    image, label = image_label
    seed = tf.random.experimental.stateless_split(seed, num=1)[0, :] # update seed

    images = []
    images.append(image)

    image = tf.image.stateless_random_brightness(
        image, max_delta=127.5, seed=seed
    )
    image = tf.clip_by_value(image, 0, 255)
    images.append(image)

    labels = []
    labels.append(label)
    labels.append(label)

    return images, labels


def apply_func_augment001():
    counter = tf.data.experimental.Counter()
    train_ds = tf.data.Dataset.zip((ds.train.unbatch(), (counter, counter)))
    train_ds = (
        train_ds
        .shuffle(1000)
        .map(func_augment001, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .unbatch()
        .shuffle(1000)
    )
    ds.update_train(train_ds)


if __name__ == '__main__':
    devs = tf.config.list_physical_devices(); print('Available devices:', devs)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    ) 


    bath_size = 32
    ds = Dataset(
         directory='dataset'
       , image_size = (224, 224)
       , image_shape = (244, 244, 3) 
       , batch_size = batch_size
       , validation_split = 0.2 
    ).load()

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
            , tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.01)
            , tf.keras.layers.experimental.preprocessing.RandomZoom(
                  height_factor=0.1
                , width_factor=0.1
            )
            , tf.keras.layers.experimental.preprocessing.RandomCrop(
                  height=200
                , width=200
            )
            #, tf.keras.layers.experimental.preprocessing.RandomContrast(
            #    factor=0.1
            #)
        ]
    )

