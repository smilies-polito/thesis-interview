import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

m = resnet18(weights=ResNet18_Weights.DEFAULT)

for p in m.parameters():
    p.requires_grad = False

m.fc = nn.Linear(m.fc.in_features, 10)
m = m.cuda()

opt = torch.optim.SGD(m.fc.parameters(), lr=1e-2, momentum=0.9)

for p in m.layer4.parameters():
    p.requires_grad = True

opt = torch.optim.SGD([
    {"params": m.layer4.parameters(), "lr": 1e-3},
    {"params": m.fc.parameters(),     "lr": 1e-2},
], momentum=0.9)

m.eval()
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    probs = torch.softmax(m(imgs.cuda()), dim=1)
    top1 = probs.argmax(dim=1)


# -----------------------------


import tensorflow as tf

base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base(x, training=False)
logits = tf.keras.layers.Dense(10)(x)

model = tf.keras.Model(inputs, logits)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

imgs = tf.random.uniform([8, 224, 224, 3])
probs = tf.nn.softmax(model(imgs, training=False), axis=-1)
top1 = tf.argmax(probs, axis=-1)
