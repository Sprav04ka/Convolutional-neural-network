import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt

# Загрузка и подготовка данных MNIST
(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()

# Изменение формы данных для CNN: (количество, высота, ширина, каналы)
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Создание модели CNN
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Компиляция модели
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Обучение модели
history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))

# Оценка точности модели
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=2)
print(f"Точность на тестовых данных: {test_accuracy:.4f}")

# Визуализация предсказания на одном из тестовых изображений
plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
predicted = model.predict(X_test[0].reshape(1, 28, 28, 1))
plt.title(
    f"Предсказанный класс: {tf.argmax(predicted[0]).numpy()}, Верный класс: {Y_test[0]}"
)
plt.show()
