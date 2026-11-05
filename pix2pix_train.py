import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from pathlib import Path
import config
from pix2pix import generator_model, generator_loss, discriminator_loss, discriminator_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _psnr_ssim(gen_output, target):
    """Return mean PSNR and SSIM for a batch; inputs are float32 in [0, 1]."""
    psnr = tf.reduce_mean(tf.image.psnr(gen_output, target, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(gen_output, target, max_val=1.0))
    return float(psnr.numpy()), float(ssim.numpy())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_pix2pix(
    train_dataset,
    val_dataset,
    output_dir,
    num_epochs=config.P2P_EPOCHS,
    batch_size=config.P2P_BATCH_SIZE,
    model_name="model_name",
    lambda_val=100,
    resume=False,
    checkpoint_dir=None,
    checkpoint_every=config.CHECKPOINT_EVERY,
    preprocess_mode=config.PREPROCESS_MODE,
):
    """Train the Pix2Pix model.

    Args:
        train_dataset: tf.data.Dataset yielding (hazy, gt) pairs in [0,1].
        val_dataset:   tf.data.Dataset yielding (hazy, gt) pairs in [0,1].
        output_dir:    Directory where the final model is saved.
        num_epochs:    Total number of epochs (counting from 0 if not resuming).
        batch_size:    Batch size (informational; dataset is pre-batched).
        model_name:    Base name (no extension) for the saved model files.
        lambda_val:    L1 weight in generator loss.
        resume:        If True, restore the latest checkpoint and continue.
        checkpoint_dir: Directory for checkpoints. Defaults to config value.
        checkpoint_every: Save a checkpoint every N epochs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = config.CHECKPOINT_DIR
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Build models ──────────────────────────────────────────────────────────
    generator     = generator_model()
    discriminator = discriminator_model()

    # ── Optimizers ────────────────────────────────────────────────────────────
    generator_optimizer     = Adam(learning_rate=2e-4, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

    # ── TF checkpoint (CG 5) ─────────────────────────────────────────────────
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        epoch=tf.Variable(0, trainable=False, dtype=tf.int64),
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, str(checkpoint_dir), max_to_keep=3
    )

    start_epoch = 0
    if resume and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        start_epoch = int(ckpt.epoch.numpy())
        print(f"[Resume] Restored checkpoint from epoch {start_epoch}: "
              f"{ckpt_manager.latest_checkpoint}")
    elif resume:
        print("[Resume] No checkpoint found; starting from scratch.")

    # ── Loss metrics ─────────────────────────────────────────────────────────
    gen_loss_metric  = Mean(name="gen_loss")
    disc_loss_metric = Mean(name="disc_loss")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        gen_loss_metric.reset_state()
        disc_loss_metric.reset_state()

        for batch in train_dataset:
            input_image, target_image = batch

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generate clear image
                gen_output = generator(input_image, training=True)

                # Discriminator patch outputs
                disc_real_output      = discriminator([input_image, target_image],  training=True)
                disc_generated_output = discriminator([input_image, gen_output],     training=True)

                # Losses
                gen_loss, gan_loss, l1_loss = generator_loss(
                    disc_generated_output, gen_output, target_image, lambda_val
                )
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            # Apply gradients
            gen_grads  = gen_tape.gradient(gen_loss,  generator.trainable_variables)
            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(
                zip(gen_grads, generator.trainable_variables)
            )
            discriminator_optimizer.apply_gradients(
                zip(disc_grads, discriminator.trainable_variables)
            )

            gen_loss_metric(gen_loss)
            disc_loss_metric(disc_loss)

        # ── Validation metrics (PSNR / SSIM) ─────────────────────────────────
        val_loss, val_psnr, val_ssim = evaluate(
            generator, val_dataset, discriminator, lambda_val
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Gen: {gen_loss_metric.result():.4f} | "
            f"Disc: {disc_loss_metric.result():.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt.epoch.assign(epoch + 1)
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == num_epochs:
            saved = ckpt_manager.save()
            print(f"  [Checkpoint] saved: {saved}")

    # ── Save final model (.keras + .h5) ───────────────────────────────────────
    keras_path = output_dir / f"{model_name}.keras"
    h5_path    = output_dir / f"{model_name}.h5"
    generator.save(str(keras_path))
    generator.save(str(h5_path))
    print(f"[Save] Generator saved to {keras_path} and {h5_path}")

    # CG3: save preprocess mode sidecar JSON
    sidecar_meta = {
        "model_name": model_name,
        "model_type": "pix2pix_generator",
        "preprocess_mode": preprocess_mode,
        "input_shape": list(config.INPUT_SHAPE),
    }
    sidecar_path = output_dir / f"{model_name}.json"
    sidecar_path.write_text(json.dumps(sidecar_meta, indent=2), encoding="utf-8")
    print(f"[Save] Preprocess mode sidecar saved to {sidecar_path}")

    return generator


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(generator, val_dataset, discriminator, lambda_val):
    """Return (mean_total_loss, mean_psnr_db, mean_ssim) on val_dataset."""
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    psnr_list, ssim_list = [], []

    for input_image, target_image in val_dataset:
        gen_output = generator(input_image, training=False)

        disc_generated_output = discriminator([input_image, gen_output],    training=False)
        disc_real_output      = discriminator([input_image, target_image],  training=False)

        gen_loss, _, _  = generator_loss(disc_generated_output, gen_output, target_image, lambda_val)
        disc_loss       = discriminator_loss(disc_real_output, disc_generated_output)
        total_loss      = gen_loss + disc_loss

        val_loss_metric.update_state(total_loss)

        p, s = _psnr_ssim(gen_output, target_image)
        psnr_list.append(p)
        ssim_list.append(s)

    mean_psnr = float(sum(psnr_list) / len(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(sum(ssim_list) / len(ssim_list)) if ssim_list else 0.0

    return float(val_loss_metric.result().numpy()), mean_psnr, mean_ssim
