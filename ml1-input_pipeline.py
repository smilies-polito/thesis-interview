import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tfms)
train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

images, labels = next(iter(train_loader))
images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)


# ---------------------------------


import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
y_train = tf.squeeze(tf.cast(y_train, tf.int32), axis=1)

MEAN = tf.constant([0.4914, 0.4822, 0.4465], tf.float32)
STD  = tf.constant([0.2470, 0.2435, 0.2616], tf.float32)

def augment(img, label):
    img = tf.image.resize_with_crop_or_pad(img, 40, 40)
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - MEAN) / STD
    return img, label

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(50_000)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(128)
            .prefetch(tf.data.AUTOTUNE))

images, labels = next(iter(train_ds))
