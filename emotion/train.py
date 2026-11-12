"""训练入口。

    uv run python -m emotion.train
"""

import argparse

from tensorflow import keras

from emotion.config import (
    BATCH_SIZE,
    CLASS_WEIGHTS,
    DOCS_DIR,
    EPOCHS,
    KERAS_PATH,
    LEARNING_RATE,
    MODELS_DIR,
    SEED,
)
from emotion.data import load_datasets
from emotion.losses import business_aware_loss
from emotion.model import build_model
from emotion.plots import plot_training_curves


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练端侧表情识别模型")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-class-weights", action="store_true", help="关闭类别加权")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, _ = load_datasets(batch_size=args.batch_size, seed=args.seed)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=business_aware_loss,
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_delta=0.0015,
            cooldown=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=35,
            min_delta=0.0015,
            restore_best_weights=True,
            start_from_epoch=35,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=None if args.no_class_weights else CLASS_WEIGHTS,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(KERAS_PATH)
    print(f"模型已保存: {KERAS_PATH}")

    out = plot_training_curves(history, DOCS_DIR / "training_curves.png", title="pupil")
    print(f"训练曲线已保存: {out}")


if __name__ == "__main__":
    main()
