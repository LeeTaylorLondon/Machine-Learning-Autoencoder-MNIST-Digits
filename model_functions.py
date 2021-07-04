import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

def build_model(summary:bool=False, test:bool=False):
    alpha = 0.05
    x_train,_,x_test,_ = load_data()
    # create encoder input layer and output layer
    encoder_input = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Flatten()(encoder_input)
    hidden_layer_1 = keras.layers.Dense(256)(x)
    activation_layer_1 = LeakyReLU(alpha=alpha)(hidden_layer_1)
    encoder_output = keras.layers.Dense(128)(activation_layer_1)
    activation_layer_2 = LeakyReLU(alpha=alpha)(encoder_output)
    # create decoder input and output layer
    hidden_layer_2 = keras.layers.Dense(256)(activation_layer_2)
    activation_layer_3 = LeakyReLU(alpha=alpha)(hidden_layer_2)
    decoder_input = keras.layers.Dense(784)(activation_layer_3)
    activation_layer_4 = LeakyReLU(alpha=alpha)(decoder_input)
    decoder_output = keras.layers.Reshape((28, 28, 1))(activation_layer_4)
    # def optimizer, model/AE and compile
    ae = keras.Model(encoder_input, decoder_output, name="AE")
    if summary: ae.summary()
    ae.compile(optimizer='adam', loss="mse",
               metrics=['accuracy'])
    # store model weights as checkpoints
    checkpoint_path = "model_weights/cp.cpkt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    # fit autoencoder, evaluate and save AE
    ae.fit(x_train, x_train, epochs=4, batch_size=8,
           callbacks=[cp_callback])
    keras.models.save_model(model=ae, filepath="model")
    if test: test_model(ae)
    return ae

def test_model(model) -> float:
    x_train,_,x_test,_ = load_data()
    rv = round(model.evaluate(x_test, x_test)[1] * 100, 4)
    print(f"\nTest Accuracy: {rv}%")
    return rv


if __name__ == '__main__':
    build_model(summary=True, test=True)


""" Compression example and stored values as real and image """
# encoder = keras.Model(encoder_io())
# example = encoder.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]
# print(example)
# print(example.shape)
# print(64/784)
# plt.imshow(example.reshape((8,8)), cmap="gray")

""" In notebook show training image then show recreated image """
# plt.imshow(x_test[0], cmap='gray')
# ae_out = ae.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]
# plt.imshow(ae_out, cmap="gray")