"""Train the U-Net or the Pix2Pix dehazing model.

    python train.py --model unet --epochs 30
    python train.py --model pix2pix --epochs 20
    python train.py --model unet --small --epochs 2      # quick debug run

After every epoch a checkpoint (weights, optimizer, epoch, early-stopping state) is
saved to outputs/checkpoints/<model>/. If a checkpoint exists, training continues from it.
Files in outputs/models/<model>/:
    <name>_best.keras + .json    model with the lowest validation loss
    <name>.keras / .h5 + .json   model at the end of training
    log.csv                      one row per epoch
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import config
from data import get_datasets
from models import discriminator as build_discriminator
from models import discriminator_loss, generator_loss, unet
from split import build_split


# Named functions (not lambdas) so Keras can save the model without warnings
def psnr(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.0)


def ssim(y_true, y_pred):
    return tf.image.ssim(y_pred, y_true, max_val=1.0)


def save_model(model, path, model_type, mode, h5=True):
    """Save path.keras (and path.h5), plus path.json so infer.py uses the same preprocessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path) + ".keras")
    if h5:
        model.save(str(path) + ".h5")
    meta = {"model_name": path.name, "model_type": model_type, "preprocess_mode": mode,
            "input_shape": list(config.INPUT_SHAPE)}
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def setup_unet():
    """Returns (model to save, objects to checkpoint, function that trains one epoch)."""
    model = unet()
    opt = Adam()  # same as optimizer="adam"
    model.compile(optimizer=opt, loss="mse", metrics=[
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.MeanMetricWrapper(fn=psnr, name="psnr"),
        tf.keras.metrics.MeanMetricWrapper(fn=ssim, name="ssim"),
    ])
    # Create the optimizer variables now, so a checkpoint can restore them
    opt.build(model.trainable_variables)

    def train_epoch(train_ds, val_ds):
        h = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2).history
        return {"train_loss": h["loss"][0], "val_loss": h["val_loss"][0],
                "val_psnr": h["val_psnr"][0], "val_ssim": h["val_ssim"][0]}

    return model, {"model": model, "optimizer": opt}, train_epoch


def setup_pix2pix(lambda_l1=100):
    """Like setup_unet. Pix2Pix val_loss is the L1 loss on validation images,
    because the GAN part of the loss does not measure image quality."""
    generator = unet()
    discriminator = build_discriminator()
    gen_opt = Adam(learning_rate=2e-4, beta_1=0.5)
    disc_opt = Adam(learning_rate=2e-4, beta_1=0.5)
    gen_opt.build(generator.trainable_variables)
    disc_opt.build(discriminator.trainable_variables)

    def train_step(hazy, clear):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake = generator(hazy, training=True)
            disc_real = discriminator([hazy, clear], training=True)
            disc_fake = discriminator([hazy, fake], training=True)
            gen_loss, _, _ = generator_loss(disc_fake, fake, clear, lambda_l1)
            disc_loss = discriminator_loss(disc_real, disc_fake)
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        return gen_loss, disc_loss

    def val_step(hazy, clear):
        fake = generator(hazy, training=False)
        return (tf.reduce_mean(tf.abs(clear - fake)),
                tf.reduce_mean(tf.image.psnr(fake, clear, max_val=1.0)),
                tf.reduce_mean(tf.image.ssim(fake, clear, max_val=1.0)))

    def train_epoch(train_ds, val_ds):
        gen_mean, disc_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        for hazy, clear in train_ds:
            gen_loss, disc_loss = train_step(hazy, clear)
            gen_mean(gen_loss)
            disc_mean(disc_loss)
        vals = [[float(v) for v in val_step(hazy, clear)] for hazy, clear in val_ds]
        l1, p, s = [sum(col) / len(vals) for col in zip(*vals)]
        return {"train_loss": float(gen_mean.result()), "disc_loss": float(disc_mean.result()),
                "val_loss": l1, "val_psnr": p, "val_ssim": s}

    objects = {"generator": generator, "discriminator": discriminator,
               "generator_optimizer": gen_opt, "discriminator_optimizer": disc_opt}
    return generator, objects, train_epoch


def read_log(log_file, last_epoch):
    """Rows of an existing log up to last_epoch (rows after the checkpoint are dropped)."""
    if last_epoch == 0 or not log_file.exists():
        return []
    with open(log_file, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if int(row["epoch"]) <= last_epoch]


def write_log(log_file, rows):
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(name, model_type, setup, train_ds, val_ds, out_dir, ckpt_dir, epochs, mode, max_hours):
    model, objects, train_epoch = setup()
    epoch_var = tf.Variable(0, dtype=tf.int64)
    best_var = tf.Variable(math.inf, dtype=tf.float64)
    bad_var = tf.Variable(0, dtype=tf.int64)  # epochs since the last improvement
    ckpt = tf.train.Checkpoint(epoch=epoch_var, best_val_loss=best_var, bad_epochs=bad_var, **objects)
    manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=2)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        print(f"[Train] Resumed from {manager.latest_checkpoint}: epoch {int(epoch_var)}, "
              f"best val loss {float(best_var):.5f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "log.csv"
    log = read_log(log_file, int(epoch_var))
    start_time = time.time()

    for epoch in range(int(epoch_var), epochs):
        if int(bad_var) >= config.EARLY_STOP_PATIENCE:
            break
        t0 = time.time()
        row = {"epoch": epoch + 1, **train_epoch(train_ds, val_ds), "seconds": round(time.time() - t0, 1)}
        print(f"[Train] Epoch {epoch + 1}/{epochs}: " + ", ".join(
            f"{k} {v:.4f}" for k, v in row.items() if k not in ("epoch", "seconds")) + f" ({row['seconds']} s)")

        if row["val_loss"] < float(best_var):
            best_var.assign(row["val_loss"])
            bad_var.assign(0)
            save_model(model, out_dir / f"{name}_best", model_type, mode, h5=False)
            print(f"[Train] New best val loss; saved {name}_best.keras")
        else:
            bad_var.assign_add(1)
        log.append(row)
        write_log(log_file, log)
        epoch_var.assign(epoch + 1)
        print(f"[Train] Checkpoint {manager.save()}")

        if int(bad_var) >= config.EARLY_STOP_PATIENCE:
            print(f"[Train] Early stop: no improvement for {config.EARLY_STOP_PATIENCE} epochs")
            break
        hours = (time.time() - start_time) / 3600
        # Stop if one more epoch would go past the limit
        if hours + row["seconds"] / 3600 > max_hours:
            print(f"[Train] Stopping: {hours:.2f} h used, limit is {max_hours} h")
            break

    save_model(model, out_dir / name, model_type, mode)
    print(f"[Train] Saved final model {out_dir / name}.keras/.h5/.json and {log_file}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a De-Haze model.")
    parser.add_argument("--model", choices=["unet", "pix2pix"], default="unet")
    parser.add_argument("--epochs", type=int, help="default: 60 for unet, 15 for pix2pix")
    parser.add_argument("--batch-size", type=int, help="default: 3")
    parser.add_argument("--small", action="store_true", help="55 outdoor pairs, for quick debug runs")
    parser.add_argument("--preprocess", choices=config.PREPROCESS_MODES, default=config.PREPROCESS_MODE)
    parser.add_argument("--max-hours", type=float, default=config.MAX_HOURS,
                        help="stop (and save) before this many hours have passed")
    parser.add_argument("--rebuild-split", action="store_true", help="rebuild splits/split_v1.json first")
    parser.add_argument("--output-dir", type=Path, help="default: outputs/models/<model>")
    parser.add_argument("--checkpoint-dir", type=Path, help="default: outputs/checkpoints/<model>")
    args = parser.parse_args()

    is_unet = args.model == "unet"
    epochs = args.epochs if args.epochs is not None else (config.UNET_EPOCHS if is_unet else config.P2P_EPOCHS)
    batch_size = args.batch_size if args.batch_size is not None else (
        config.UNET_BATCH_SIZE if is_unet else config.P2P_BATCH_SIZE)
    out_dir = args.output_dir or config.MODELS_DIR / args.model
    ckpt_dir = args.checkpoint_dir or config.CHECKPOINT_DIR / args.model

    print(f"[Train] {args.model}, {epochs} epochs, batch {batch_size}, preprocess {args.preprocess}, "
          f"small={args.small}, max {args.max_hours} h, output {out_dir}, checkpoints {ckpt_dir}")
    print(f"[Train] GPUs: {tf.config.list_physical_devices('GPU') or 'none, using CPU'}")

    if args.rebuild_split:
        build_split(rebuild=True)
    train_ds, val_ds = get_datasets(batch_size, args.preprocess, args.small)

    if is_unet:
        train("unet", "unet", setup_unet, train_ds, val_ds, out_dir, ckpt_dir,
              epochs, args.preprocess, args.max_hours)
    else:
        train("p2p", "pix2pix_generator", setup_pix2pix, train_ds, val_ds, out_dir, ckpt_dir,
              epochs, args.preprocess, args.max_hours)


if __name__ == "__main__":
    main()
