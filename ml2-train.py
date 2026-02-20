import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

model = nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, 10)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10_000)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

model.train()
images, labels = next(iter(train_loader))
images, labels = images.cuda(), labels.cuda()

opt.zero_grad(set_to_none=True)
with autocast(dtype=torch.float16):
    logits = model(images)      # raw scores
    loss = crit(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(opt)  # important before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(opt)
scaler.update()
sched.step()


# -------------------------


import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, dtype="float32"),  # keep logits in fp32 for stability
])

base_opt = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-2)
opt = mixed_precision.LossScaleOptimizer(base_opt)
crit = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
lr_sched = tf.keras.optimizers.schedules.CosineDecay(3e-4, decay_steps=10_000)

images, labels = next(iter(train_ds))
labels_oh = tf.one_hot(labels, depth=10)

with tf.GradientTape() as tape:
    logits = model(images, training=True)
    loss = crit(labels_oh, logits)
    scaled_loss = opt.get_scaled_loss(loss)

scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
grads = opt.get_unscaled_gradients(scaled_grads)
grads, _ = tf.clip_by_global_norm(grads, 1.0)

opt.learning_rate = lr_sched(opt.iterations)
opt.apply_gradients(zip(grads, model.trainable_variables))
