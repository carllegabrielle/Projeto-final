import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1) Definições gerais
BASE_DATA  = 'data'           # pasta com subpastas train/ e test/
MODEL_OUT  = 'models/model.h5'
IMG_SIZE   = (64, 64)
BATCH_SIZE = 32
EPOCHS     = 30

# 2) Função para remover linhas do grid
def remove_grid_lines(gray: np.ndarray) -> np.ndarray:
    """
    Recebe array 2D uint8 (0–255), detecta linhas horizontais/verticais finas
    e retorna a imagem com o grid removido.
    """
    h, w = gray.shape
    # binariza invertido p/ destacar o grid
    _, bw_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # extrai horizontais
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (w//10, 1))
    horizontals = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, hor_k)

    # extrai verticais
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//10))
    verticals = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, ver_k)

    # remove grid
    grid  = cv2.add(horizontals, verticals)
    clean = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(grid))
    return clean

# 3) Função de pré-processamento para o ImageDataGenerator
def preprocess_and_clean(img: np.ndarray) -> np.ndarray:
    """
    Recebe img como numpy array HxWx1 uint8 (0–255),
    remove grid, normaliza e retorna float32 HxWx1 (0–1).
    """
    # squeeze pra 2D
    gray = img.squeeze().astype(np.uint8)
    clean = remove_grid_lines(gray)

    # redimensiona caso necessário
    if clean.shape != IMG_SIZE:
        clean = cv2.resize(clean, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # normaliza para [0,1] e devolve com shape (64,64,1)
    arr = clean.astype(np.float32) / 255.0
    return arr.reshape(*IMG_SIZE, 1)

# 4) Configura ImageDataGenerators
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_and_clean,
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    zoom_range=0.05,
    shear_range=2
).flow_from_directory(
    os.path.join(BASE_DATA, 'train'),
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_and_clean
).flow_from_directory(
    os.path.join(BASE_DATA, 'test'),
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=False,
)

num_classes = len(train_gen.class_indices)

# 5) Definição da arquitetura (baseline simples)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 1)),
    layers.MaxPool2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6) Callbacks para salvar o melhor modelo
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# 7) Treinar
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 8) Avaliação final
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"Teste → Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 9) O modelo final foi salvo em models/model.h5
