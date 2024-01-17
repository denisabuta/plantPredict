from tf_model import data_preprocessing, build_model, train_model, use_pretrained_net, model_predictions
import tensorflow as tf

use_data_dir = True
data_dir = 'Data/Plants'

model_name = 'pretrained.h5'

train = False
load = True
predict = True

#pentru a suprascrie modelul propriu cu cel preantrenat inceptionv3
pretrained = True

if __name__ == '__main__':
    directory = ''
    model = tf.keras.models.Sequential()

    if use_data_dir:
        directory = data_dir

    train_ds, val_ds, img_dims = data_preprocessing(validation_split=0.2, img_dims=256, data_dir=directory)
    class_names = train_ds.class_names

    if train:
        model = build_model(output_dims=len(class_names), input_dims=img_dims)
        if pretrained:
            model = use_pretrained_net(output_dims=len(class_names), input_dims=img_dims)
        train_model(model=model, train_ds=train_ds, val_ds=val_ds, epochs=15, model_name=model_name)
    elif load:
        model = tf.keras.models.load_model(model_name)

    if predict:
        if use_data_dir:
            model_predictions(model=model, img_dims=img_dims, class_names=class_names, pretrained=pretrained,
                              folder='Plants')