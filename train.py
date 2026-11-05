"""Train the U-Net or the Pix2Pix dehazing model.

    python train.py --model unet --epochs 60
    python train.py --model pix2pix --epochs 15 --resume
    python train.py --model unet --small --epochs 1      # quick debug run
"""
import argparse
import json
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


def save_model(model, out_dir, name, model_type, mode):
    """Save .keras and .h5, plus a .json sidecar so infer.py uses the same preprocessing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / f"{name}.keras"))
    model.save(str(out_dir / f"{name}.h5"))
    meta = {"model_name": name, "model_type": model_type, "preprocess_mode": mode,
            "input_shape": list(config.INPUT_SHAPE)}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Save] {out_dir / name}.keras / .h5 / .json")


def train_unet(train_ds, val_ds, out_dir, epochs, mode):
    model = unet()
    model.compile(optimizer="adam", loss="mse", metrics=[
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.MeanMetricWrapper(fn=psnr, name="psnr"),
        tf.keras.metrics.MeanMetricWrapper(fn=ssim, name="ssim"),
    ])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    save_model(model, out_dir, "unet", "unet", mode)
    return model


def evaluate_pix2pix(generator, discriminator, val_ds, lambda_l1):
    """Mean (total loss, PSNR, SSIM) over the validation batches."""
    losses, psnrs, ssims = [], [], []
    for hazy, clear in val_ds:
        fake = generator(hazy, training=False)
        disc_fake = discriminator([hazy, fake], training=False)
        disc_real = discriminator([hazy, clear], training=False)
        gen_loss, _, _ = generator_loss(disc_fake, fake, clear, lambda_l1)
        losses.append(float(gen_loss + discriminator_loss(disc_real, disc_fake)))
        psnrs.append(float(tf.reduce_mean(tf.image.psnr(fake, clear, max_val=1.0))))
        ssims.append(float(tf.reduce_mean(tf.image.ssim(fake, clear, max_val=1.0))))
    n = max(len(losses), 1)
    return sum(losses) / n, sum(psnrs) / n, sum(ssims) / n


def train_pix2pix(train_ds, val_ds, out_dir, epochs, mode, resume=False,
                  ckpt_dir=config.CHECKPOINT_DIR, lambda_l1=100):
    generator = unet()
    discriminator = build_discriminator()
    gen_opt = Adam(learning_rate=2e-4, beta_1=0.5)
    disc_opt = Adam(learning_rate=2e-4, beta_1=0.5)

    # The checkpoint also stores the optimizers and the epoch, so --resume continues exactly
    ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator,
                               generator_optimizer=gen_opt, discriminator_optimizer=disc_opt,
                               epoch=tf.Variable(0, trainable=False, dtype=tf.int64))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=3)
    start_epoch = 0
    if resume and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        start_epoch = int(ckpt.epoch.numpy())
        print(f"[Resume] Restored {manager.latest_checkpoint} (epoch {start_epoch})")
    elif resume:
        print("[Resume] No checkpoint found; starting from scratch.")

    gen_mean = tf.keras.metrics.Mean()
    disc_mean = tf.keras.metrics.Mean()
    for epoch in range(start_epoch, epochs):
        gen_mean.reset_state()
        disc_mean.reset_state()
        for hazy, clear in train_ds:
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
            gen_mean(gen_loss)
            disc_mean(disc_loss)

        val_loss, val_psnr, val_ssim = evaluate_pix2pix(generator, discriminator, val_ds, lambda_l1)
        print(f"Epoch {epoch + 1}/{epochs} | Gen: {gen_mean.result():.4f} | Disc: {disc_mean.result():.4f} | "
              f"Val loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        ckpt.epoch.assign(epoch + 1)
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0 or epoch + 1 == epochs:
            print(f"  [Checkpoint] {manager.save()}")

    save_model(generator, out_dir, "p2p", "pix2pix_generator", mode)
    return generator


def main():
    parser = argparse.ArgumentParser(description="Train a De-Haze model.")
    parser.add_argument("--model", choices=["unet", "pix2pix"], default="unet")
    parser.add_argument("--epochs", type=int, help="default: 60 for unet, 15 for pix2pix")
    parser.add_argument("--batch-size", type=int, help="default: 3")
    parser.add_argument("--small", action="store_true", help="55 outdoor pairs, for quick debug runs")
    parser.add_argument("--resume", action="store_true", help="Pix2Pix: continue from the last checkpoint")
    parser.add_argument("--preprocess", choices=config.PREPROCESS_MODES, default=config.PREPROCESS_MODE)
    parser.add_argument("--rebuild-split", action="store_true", help="rebuild splits/split_v1.json first")
    parser.add_argument("--output-dir", type=Path, help="default: outputs/models/<model>")
    args = parser.parse_args()

    is_unet = args.model == "unet"
    epochs = args.epochs if args.epochs is not None else (config.UNET_EPOCHS if is_unet else config.P2P_EPOCHS)
    batch_size = args.batch_size if args.batch_size is not None else (
        config.UNET_BATCH_SIZE if is_unet else config.P2P_BATCH_SIZE)
    out_dir = args.output_dir or config.MODELS_DIR / args.model

    print(f"[Train] {args.model}, {epochs} epochs, batch {batch_size}, preprocess {args.preprocess}, "
          f"small={args.small}, resume={args.resume}, output {out_dir}")
    print(f"[Train] GPUs: {tf.config.list_physical_devices('GPU') or 'none, using CPU'}")

    if args.rebuild_split:
        build_split(rebuild=True)
    train_ds, val_ds = get_datasets(batch_size, args.preprocess, args.small)

    if is_unet:
        if args.resume:
            print("[Train] --resume only applies to pix2pix; training U-Net from scratch.")
        train_unet(train_ds, val_ds, out_dir, epochs, args.preprocess)
    else:
        train_pix2pix(train_ds, val_ds, out_dir, epochs, args.preprocess, resume=args.resume)


if __name__ == "__main__":
    main()
