import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🔁 Навчання моделі...")
history = model.fit(x_train, y_train_cat, epochs=10, batch_size=64,
                    validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n✅ Точність на тестових даних: {test_acc:.4f}")

model.save("fashion_cnn_model.h5")
print("💾 Модель збережена як fashion_cnn_model.h5")

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Точність моделі')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Похибка моделі')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("fashion_training_plots.png")
    plt.show()
    print("📈 Графіки збережено як fashion_training_plots.png")

plot_history(history)


class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Файл '{image_path}' не знайдено.")
        return

    model = tf.keras.models.load_model("fashion_cnn_model.h5")

    img = Image.open(image_path).convert("L") 
    img = ImageOps.invert(img) 
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\n🔍 Клас: {predicted_class} → {class_names[predicted_class]} ({confidence:.2%} впевненість)")


predict_image("images/image.jpg")