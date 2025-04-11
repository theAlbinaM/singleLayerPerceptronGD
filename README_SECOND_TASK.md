### Подготовка обучающей выборки
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0  # Преобразуем изображения в вектор и нормализуем
x_test = x_test.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train) # One-hot encoding выходных значений
y_test = to_categorical(y_test)
```

### Создание и обучение модели персептрона с заданной скоростью обучения
```python
input_size = 28*28  # Количество входов
output_size = 10      # Количество выходов
learning_rate = 0.5
epochs = 25

# Списки для хранения ошибок и точности на каждой эпохе
losses = []
accuracies = []
```

### Функция активации Softmax
```python
def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
  return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### Шаг 1: Инициализация весов и смещений случайными малыми значениями
```python
np.random.seed(42)
weights = np.random.rand(input_size, output_size) * 0.01  # Случайные малые веса
biases = np.random.uniform(-1,1,output_size)  # Смещения инициализируем малыми значениями
```

### Шаг 2-5: Обучение методом положительного и отрицательного подкрепления
```python
def predict(inputs):
    activation = np.dot(inputs, weights) + biases # Вычисление активации
    return softmax(activation)

def train(training_data, training_labels):
    for _ in range(epochs):
      predictions = predict(training_data)

      # Обновляем веса
      for i in range(training_data.shape[0]):
        predicted_class = np.argmax(predictions[i])
        true_class = np.argmax(training_labels[i])

        if predicted_class == true_class:
            # Положительное подкрепление: увеличиваем веса для правильного класса
            weights[:, predicted_class] += training_data[i] * learning_rate  # Увеличиваем вес
            biases[predicted_class] += learning_rate
        else:
            # Отрицательное подкрепление: уменьшаем веса для неправильного класса
            weights[:, predicted_class] -= training_data[i] * learning_rate  # Уменьшаем вес
            biases[predicted_class] -= learning_rate
            weights[:, true_class] += training_data[i] * learning_rate  # Увеличиваем вес для правильного класса
            biases[true_class] += learning_rate

      # Вычисление потерь и точности
      predictions = predict(training_data)
      loss = np.mean(np.square(predictions - training_labels))  # Среднеквадратичная ошибка
      accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(training_labels, axis=1))

      losses.append(loss)
      accuracies.append(accuracy)
      print(f"Epoch {_+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
```

### Обучение модели
```python
train(x_train, y_train)
```

### Оценка на тестовых данных
```python
predictions = predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy_test = np.mean(predicted_classes == true_classes)
print(f"Точность на тестовых данных: {accuracy_test * 100:.2f}%")
```

### Графики ошибки и точности
```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, color = "red")
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies, color = "green")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
```

### Результат
<img width="762" alt="Снимок экрана 2025-04-11 в 10 37 10" src="https://github.com/user-attachments/assets/e89cb0d4-afb6-4c88-9d45-9c13ce046978" />
