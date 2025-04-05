### Шаг 1: Подготовка обучающей выборки, нормализация данных и преобразование в векторы

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### Шаг 2-3: Задание параметров обучения, инициализация весов и смещений случайными малыми значениями

```python
input_size = 28*28 
output_size = 10 
learning_rate = 0.2
epochs = 10
batch_size = 128

np.random.seed(42)
W = np.random.randn(input_size, output_size) * 0.01  # Случайные малые веса
B = np.zeros((1, output_size))  # Смещения инициализируем нулями
```

### Шаг 5: вычисление выхода нейронов: функция активации softmax

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Создание списков для хранения ошибок и точности на каждой эпохе
losses = []
accuracies = []
```

### Шаг 4-9: Обучение методом градиентного спуска

```python
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        # Шаг 4: Берем батч из обучающей выборки
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Шаг 5: Вычисляем взвешенную сумму и softmax-выход
        net = np.dot(X_batch, W) + B  # Взвешенная сумма входов
        predictions = softmax(net)  # Получаем вероятности классов

        # Шаг 6: Вычисление ошибки (функция потерь)
        loss = -np.mean(y_batch * np.log(predictions))
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))  # Точность предсказаний

        # Шаг 7: Вычисление градиентов
        d_output = (predictions - y_batch) / batch_size  # Градиент softmax
        dW = np.dot(X_batch.T, d_output)  # Градиент по весам
        dB = np.sum(d_output, axis=0, keepdims=True)  # Градиент по смещениям

        # Шаг 9: Обновление весов методом градиентного спуска
        W -= learning_rate * dW
        B -= learning_rate * dB

    # Добавляем метрики каждой эпохи
    losses.append(loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
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

### Оценка на тестовых данных

```python
logits_test = np.dot(X_test, W) + B
predictions_test = softmax(logits_test)
test_accuracy = np.mean(np.argmax(predictions_test, axis=1) == np.argmax(y_test, axis=1))

print(f"Точность на тестовых данных: {test_accuracy:.4f}")
```

### Результат

<img width="1023" alt="Снимок экрана 2025-04-05 в 08 26 37" src="https://github.com/user-attachments/assets/3cd4ef5f-463a-4402-8ec0-4c1f741bb683" />



