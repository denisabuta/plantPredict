import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import numpy as np

def data_preprocessing(validation_split=0.2, batch_size=32, img_dims=180, data_dir='Data/Plants'):
    #funcție de preprocesare a datelor pentru imagini - generează seturi de date pentru antrenament și validare
    data_dir = data_dir
    batch_size = batch_size
    #dimensiunea fiecărui lot de imagini care va fi folosit în timpul antrenării și validării rețelei neurale
    img_height = img_dims
    img_width = img_dims

    #stocarea setului de date folosit pentru antrenare
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size, )

    #stocarea setului de date folosit pentru validare
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds, img_dims

def build_model(output_dims, input_dims):
    #crearea modelului de rețea neurală convoluțională
    #adăugarea secvențială a straturilor de rețea neurală
    model = Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(input_dims,
                                                                  input_dims,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dims)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds, epochs, model_name):
    epochs = epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(model_name + '_plot.png')
    plt.clf()
    model.save(model_name)

def use_pretrained_net(output_dims, input_dims):
    #model secvențial pentru adăugarea straturilor rețelei neurale
    inception_model = Sequential()
    #incarcarea modelul pre-antrenat InceptionV3
    inception_base = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                    input_shape=(input_dims, input_dims, 3))
    inception_model.add(inception_base)
    inception_model.add(layers.GlobalAveragePooling2D())
    inception_model.add(layers.Dense(units=output_dims, activation='softmax'))
    inception_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return inception_model

def clear_folder(path='predictions/'):
    #functie de stergere a folderului de predictie folosita la fiecare rulare noua
    for file_name in os.listdir(path):
        file = path + file_name
        if os.path.isfile(file):
            os.remove(file)

def model_predictions(model, img_dims, class_names, pretrained, folder):
    #functie pentru predictia carei specii de plante apartine frunza
    #prelucrare directoare pentru rularea predictiei
    dir_test = 'test/'
    dir_test = dir_test + folder + '/'
    img_location = dir_test
    dir_test = os.listdir(dir_test)
    images = list(dir_test)
    save_predictions_path = 'predictions/'
    clear_folder(path=save_predictions_path)
    # incarcam imaginile, redimensionam si transformam in numpy array
    for image in images:
        path = img_location + image
        img = tf.keras.preprocessing.image.load_img(path, target_size=(img_dims, img_dims))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)[0]
        #daca modelul nu este preantrenat => folosim softmax pe imagini
        if not pretrained:
            predictions = tf.nn.softmax(predictions)
        #selectam primele 3 clase cu cea mai mare probabilitate
        max_predictions = np.argpartition(predictions, -3)[-3:]
        max_predictions_sorted = max_predictions[np.argsort(predictions[max_predictions])[::-1]]
        
        title_str = ""
        for i, class_index in enumerate(max_predictions_sorted):
            title_str += f'{class_names[class_index]} {format((predictions[class_index]), ".3f")}%  '
            if i == 2:
                break
        plt.imshow(img)
        plt.title(title_str, fontsize=8)
        plt.savefig(os.path.join(save_predictions_path, str(len(os.listdir(save_predictions_path))) + '.png'))