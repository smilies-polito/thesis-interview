import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# [PT-1] deterministic split between train and validation
g = torch.Generator().manual_seed(7)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

eval_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

raw_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=None)
perm = torch.randperm(len(raw_ds), generator=g)
split = int(0.9 * len(raw_ds))
train_idx = perm[:split].tolist()
val_idx = perm[split:].tolist()

# [PT-2] two dataset views over the same files, each with its own transform
train_ds_all = datasets.CIFAR10(root="data", train=True, download=False, transform=train_tfms)
val_ds_all = datasets.CIFAR10(root="data", train=True, download=False, transform=eval_tfms)

train_ds = Subset(train_ds_all, train_idx)
val_ds = Subset(val_ds_all, val_idx)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

train_images, train_labels = next(iter(train_loader))
val_images, val_labels = next(iter(val_loader))

train_images = train_images.cuda(non_blocking=True)
train_labels = train_labels.cuda(non_blocking=True)
val_images = val_images.cuda(non_blocking=True)
val_labels = val_labels.cuda(non_blocking=True)


# ---------------------------------


import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
y_train = tf.squeeze(tf.cast(y_train, tf.int32), axis=1)

MEAN = tf.constant([0.4914, 0.4822, 0.4465], tf.float32)
STD = tf.constant([0.2470, 0.2435, 0.2616], tf.float32)


def normalize(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - MEAN) / STD
    return img, label


def augment(img, label):
    img = tf.image.resize_with_crop_or_pad(img, 40, 40)
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)
    return normalize(img, label)


# [TF-1] deterministic split between train and validation
n = tf.shape(x_train)[0]
idx = tf.random.experimental.stateless_shuffle(tf.range(n), seed=[7, 11])
split = tf.cast(tf.round(tf.cast(n, tf.float32) * 0.9), tf.int32)

train_idx = idx[:split]
val_idx = idx[split:]

x_tr = tf.gather(x_train, train_idx)
y_tr = tf.gather(y_train, train_idx)
x_val = tf.gather(x_train, val_idx)
y_val = tf.gather(y_train, val_idx)

train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
            .shuffle(tf.shape(x_tr)[0])
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(128, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
          .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(256)
          .cache()
          .prefetch(tf.data.AUTOTUNE))

train_images, train_labels = next(iter(train_ds))
val_images, val_labels = next(iter(val_ds))

