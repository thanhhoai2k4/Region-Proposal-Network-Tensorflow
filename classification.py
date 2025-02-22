import tensorflow as tf

directory = "data_training_classification/images"

def Rescaling(image, label):
    return image / 255.0, label

def onehot(image ,label):
    C = tf.constant(4, name="C")
    label = tf.one_hot(
        label, C, on_value=1.0, off_value=0.0, axis=-1)
    return image, label

def data():

    dataset_train, dataset_val = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(100, 100),  # Kích thước mà hình ảnh sẽ được resize
        batch_size=64,          # Số lượng hình ảnh trong mỗi batch
        shuffle=True,            # Xáo trộn dữ liệu
        validation_split = 0.2,
        subset="both",
        seed=100
    )

    dataset_train = dataset_train.map(Rescaling)
    dataset_train = dataset_train.map(onehot)

    dataset_val = dataset_val.map(Rescaling)
    dataset_val = dataset_val.map(onehot)

    return dataset_train, dataset_val



def model(input_shape: tuple):
    inputs = tf.keras.Input(input_shape)
    layer1 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation="relu"
    )(inputs)
    normal = tf.keras.layers.BatchNormalization()(layer1)
    layer2 = tf.keras.layers.MaxPooling2D((2, 2))(normal)
    layer3 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size = (3,3),
        activation="relu"
    )(layer2)
    normal = tf.keras.layers.BatchNormalization()(layer3)
    layer4 = tf.keras.layers.MaxPooling2D((2, 2))(normal)

    flatten = tf.keras.layers.Flatten()(layer4)
    x = tf.keras.layers.Dense(64, activation="relu")(flatten)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(4, activation="softmax")(x)




    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

def train_model(model, dataset):

    data_train,data_val = dataset

    data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
    data_val = data_val.prefetch(tf.data.experimental.AUTOTUNE)

    optimize = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
    model.compile(
        optimizer=optimize,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(data_train, epochs=30, verbose=1,validation_data=data_val)
    model.save("my_model.h5")


model = model((100,100,3))
model.load_weights("my_model.h5")
train, val = data()
train_model(model, (train, val))