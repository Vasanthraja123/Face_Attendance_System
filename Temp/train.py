import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define constants
input_shape = (128, 128, 3)
num_classes = 3
batch_size = 32
epochs = 30  # More epochs for better learning
learning_rate = 0.001

# Data Augmentation to Improve Generalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    'Dataset.zip/train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'Dataset.zip/test',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load Pre-trained MobileNetV2 Model (Transfer Learning)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

# Freeze base model layers
base_model.trainable = False  

# Build Custom Model on top of MobileNetV2
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks to Improve Training
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),  # Reduce LR when stuck
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)  # Stop early if overfitting
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks
)

# Save the trained model
model.save('advanced_face_recognition.h5')
