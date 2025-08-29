import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
)
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ----------------------------
# Helper: build model
# ----------------------------
def build_model(name, img_size, num_classes):
    if name == "scratch":
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        base = None
        if name == "vgg16":
            base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
        elif name == "resnet50":
            base = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
        elif name == "mobilenetv2":
            base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
        elif name == "inceptionv3":
            base = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
        elif name == "efficientnetb0":
            base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))

        base.trainable = False
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax")
        ])
    return model

# ----------------------------
# Training Function
# ----------------------------
def train(args):
    img_size = (args.img_size[0], args.img_size[1])

    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        subset="validation"
    )

    num_classes = len(train_gen.class_indices)

    all_models = args.models if args.models else ["scratch", "vgg16", "resnet50", "mobilenetv2", "inceptionv3", "efficientnetb0"]

    for model_name in all_models:
        print(f"\nTraining {model_name} ...")

        model = build_model(model_name, img_size, num_classes)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Stage 1
        history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_stage1)

        # Stage 2 (fine-tune if transfer learning)
        if model_name != "scratch":
            print("Fine-tuning last layers...")
            model.layers[0].trainable = True
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_stage2)

        # Save model
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{model_name}_best.h5")

        # Save metrics
        val_preds = np.argmax(model.predict(val_gen), axis=1)
        val_true = val_gen.classes
        report = classification_report(val_true, val_preds, target_names=list(val_gen.class_indices.keys()), output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        os.makedirs("reports", exist_ok=True)
        df_report.to_csv(f"reports/{model_name}_metrics.csv")

        # Confusion Matrix
        cm = confusion_matrix(val_true, val_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(f"reports/{model_name}_confusion_matrix.png")
        plt.close()

        # History plots
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.legend()
        plt.title(f"{model_name} Accuracy")
        plt.savefig(f"reports/{model_name}_history_acc.png")
        plt.close()

        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.title(f"{model_name} Loss")
        plt.savefig(f"reports/{model_name}_history_loss.png")
        plt.close()

        print(f"Finished training {model_name}. Model and reports saved.")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--img_size", type=int, nargs=2, default=[224,224])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    train(args)
