import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32 * 32 * 3, 512),
    nn.GELU(),
    nn.LayerNorm(512),
    nn.Linear(512, 10),
).cuda()

# [PT-1] exclude bias/norm params from weight decay
decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim == 1 or name.endswith(".bias"):
        no_decay.append(param)
    else:
        decay.append(param)

opt = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 1e-2},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
)

total_steps = 10_000
warmup_steps = 500


def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

accum_steps = 4
model.train()
opt.zero_grad(set_to_none=True)

# [PT-2] gradient accumulation over multiple micro-batches
it = iter(train_loader)
for _ in range(accum_steps):
    images, labels = next(it)
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    with autocast(dtype=torch.float16):
        logits = model(images)
        loss = crit(logits, labels) / accum_steps

    scaler.scale(loss).backward()

scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(opt)
scaler.update()
opt.zero_grad(set_to_none=True)
sched.step()


# -------------------------


import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="gelu"),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(10, dtype="float32"),
])

base_opt = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.95)
opt = mixed_precision.LossScaleOptimizer(base_opt)
crit = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

total_steps = 10_000
warmup_steps = 500
peak_lr = 3e-4
weight_decay = 1e-2
accum_steps = 4


def warmup_cosine(step):
    step = tf.cast(step, tf.float32)
    warmup = tf.minimum(1.0, (step + 1.0) / tf.cast(warmup_steps, tf.float32))
    progress = tf.clip_by_value(
        (step - tf.cast(warmup_steps, tf.float32))
        / tf.cast(max(1, total_steps - warmup_steps), tf.float32),
        0.0,
        1.0,
    )
    cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * progress))
    return peak_lr * tf.where(step < warmup_steps, warmup, cosine)


train_iter = iter(train_ds)
accum_grads = [tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]

# [TF-1] gradient accumulation over multiple micro-batches
for _ in range(accum_steps):
    images, labels = next(train_iter)

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        ce = crit(labels, logits)
        wd = tf.add_n([
            tf.nn.l2_loss(v) for v in model.trainable_variables if "kernel" in v.name
        ]) * weight_decay
        loss = (ce + wd) / accum_steps
        scaled_loss = opt.get_scaled_loss(loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)
    for slot, grad in zip(accum_grads, grads):
        slot.assign_add(grad)

grads = [slot.read_value() for slot in accum_grads]
grads, _ = tf.clip_by_global_norm(grads, 1.0)
base_opt.learning_rate = warmup_cosine(opt.iterations)
opt.apply_gradients(zip(grads, model.trainable_variables))

for slot in accum_grads:
    slot.assign(tf.zeros_like(slot))

