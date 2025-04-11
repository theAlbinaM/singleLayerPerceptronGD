### Подготовка обучающей и тестовой выборок
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

### 1-ая архитектура
```python
model = Sequential()
model.add(Dense(10, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
```

### Результат
<img width="909" alt="Снимок экрана 2025-04-11 в 19 46 45" src="https://github.com/user-attachments/assets/eae32035-9d7a-4875-9832-078968a8d873" />

### 2-ая архитектура
```python
model2 = Sequential()
model2.add(Dense( 50, input_dim=784, activation='relu'))
model2.add(Dense(10, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model2.evaluate(x_test, y_test)
print(test_acc)
```

### Результат
<img width="905" alt="Снимок экрана 2025-04-11 в 19 47 58" src="https://github.com/user-attachments/assets/ffb2d844-15b6-415d-8b3b-88573b076438" />

### 3-я архитектура
```python
model3 = Sequential()
model3.add(Dense(50, input_dim=784, activation='relu'))
model3.add(Dense(50, activation='relu'))
model3.add(Dense(10, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model3.evaluate(x_test, y_test)
print(test_acc)
```

### Результат
<img width="909" alt="Снимок экрана 2025-04-11 в 19 48 57" src="https://github.com/user-attachments/assets/66cfb1c1-751b-4db2-8e47-e9e898a134ca" />

### 4-ая архитектура
```python
# Загружаем данные
(x_train_cnn, y_train_cnn), (x_test_cnn, y_test_cnn) = mnist.load_data()
x_train_cnn = x_train_cnn[:,:,:,np.newaxis] / 255.0
x_test_cnn = x_test_cnn[:,:,:,np.newaxis] / 255.0
y_train_cnn = to_categorical(y_train_cnn)
y_test_cnn = to_categorical(y_test_cnn)

# Опишем архитектуру сети 4
model4 = Sequential()
model4.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28, 1)))
model4.add(MaxPooling2D(pool_size=2))
model4.add(Flatten())
model4.add(Dense(10, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model4.fit(x_train_cnn, y_train_cnn, epochs=10, validation_split=0.1)

_, test_acc = model4.evaluate(x_test_cnn, y_test_cnn)
print(test_acc)
```

### Результат
<img width="922" alt="Снимок экрана 2025-04-11 в 19 50 04" src="https://github.com/user-attachments/assets/3c55a00d-090d-4564-91a9-e1f64d14f030" />

### Проведите сравнение работы четырех архитектур.
#### Какая из архитектур показывает лучший результат и почему?
Лучший результат у CNN (4 модель), так как она может «видеть» подмножество данных. С помощью нее можно обнаружить образ на изображениях лучше, чем при работе с персептроном.

#### Как можно улучшить результаты каждой из архитектур?
Добавить дополнительные слои, подобрать оптимальные параметры и гиперпараметры обучения.

### По вариантам (в соответствии с номером в списке) спроектируйте архитектуру, реализуйте алгоритм корректировки синаптических весов с помощью алгоритма обратного распространения ошибки, используя в качестве функции активации логистический сигмоид 𝑓(𝑛𝑒𝑡) = 1/ (1+exp(-net))
***Комментарий: Обратное распространение ошибки реализовано автоматически при вызове метода fit()***

#### Вариант 8
- Архитектура 3-3-2
- Скорость обучения 0.25
- Входной вектор X={0.4;-0.8;0.2}

***Комментарий: Не может быть использован, так как входные данные преобразуются в вектор с признаками равным количеству 28x28 при использовании mnist***

- Матрицы синапсов 1 и 2 слоя:
Начальные значения весов для первого слоя взять произвольным образом из интервала [-0.3 0.3]

***Комментарий: Данные веса будут применены ко всем слоям***

- W2={0.4 -0.7; 1.2 0.6; 0.1 0.5; -1.4 0.5}

***Комментарий: Не могут быть использованы, так как слоя из 4 нейронов в заданных архитектурах нет***

- Эталонный выход Y={0.5;0.3}

***Комментарий: Не может быть использован, так как решается задача классификации с 10 классами***

### Новая архитектура
```python
# Инициализируем веса для каждого слоя
first_weights = np.random.uniform(-0.3, 0.3, (28*28, 50))
weights_50_50 = np.random.uniform(-0.3, 0.3, (50, 50))
weights_50_10 = np.random.uniform(-0.3, 0.3, (50, 10))
weights_10_50 = np.random.uniform(-0.3, 0.3, (10, 50))

# Опишем архитектуру новой сети
model__new = Sequential()
model__new.add(Dense(50, kernel_initializer = first_weights, input_dim=28*28, activation='sigmoid'))
model__new.add(Dense(50, kernel_initializer = weights_50_50, activation='sigmoid'))
model__new.add(Dense(10, kernel_initializer = weights_50_10, activation='sigmoid'))

model__new.add(Dense(50, kernel_initializer = weights_10_50, activation='sigmoid'))
model__new.add(Dense(50, kernel_initializer = weights_50_50, activation='sigmoid'))
model__new.add(Dense(10, kernel_initializer = weights_50_10, activation='sigmoid'))

model__new.add(Dense(50, kernel_initializer = weights_10_50, activation='sigmoid'))
model__new.add(Dense(50, kernel_initializer = weights_50_50, activation='sigmoid'))
model__new.add(Dense(10, kernel_initializer = weights_50_10, activation='sigmoid'))

model__new.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.25), metrics=['accuracy'])
model__new.fit(x_train, y_train, epochs=10, validation_split=0.1)

_, test_acc = model__new.evaluate(x_test, y_test)
print(test_acc)
```

### Результат
<img width="913" alt="Снимок экрана 2025-04-11 в 19 50 52" src="https://github.com/user-attachments/assets/4add1ef5-e972-4507-b1a2-90fe7bedfdae" />


### Вывод
Точность модели крайне мала на 10 эпохах, возможные причины этого:
- неудачная архитектура сети (50-50-10...)
- слишком большое значение скорости обучения
- неподходящая функция активации
