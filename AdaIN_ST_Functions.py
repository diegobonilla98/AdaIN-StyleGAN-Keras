import DataLoader
from increase_memory_alloc import K, tf
from BoniDL.losses import image_euclidean_loss
from tensorflow.keras.applications import vgg19
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Lambda, Layer, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")


class SpatialReflectionPadding(Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, xx):
        return tf.pad(xx, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), mode="REFLECT")


class StyleTransfer:
    def __init__(self, trained_weights=None):
        self.IMAGE_SIZE = 512
        self.dl = DataLoader.DataLoader(self.IMAGE_SIZE, 2)

        self.test_contents = np.array([self.dl._load_image(p) for p in glob.glob('./inputs/images/*')])
        self.test_styles = np.array([self.dl._load_image(p) for p in glob.glob('./inputs/styles/*')])

        self.vgg_backbone = vgg19.VGG19(weights='imagenet', include_top=False)
        self.vgg_backbone.trainable = False
        for l in self.vgg_backbone.layers:
            l.trainable = False
        self.first_feat_extractor = Model(inputs=self.vgg_backbone.input,
                                          outputs=self.vgg_backbone.get_layer('block4_conv1').output,  # block5_conv2
                                          name='feature_extractor')
        self.first_feat_extractor.trainable = False

        self.decoder = self.get_decoder()
        if trained_weights is not None:
            self.decoder.load_weights(trained_weights)
        self.encoder, self.full_model = self.create_full_model()
        self.model_train = self.build_model()

    def style_loss(self, y_pred, s):
        # layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',  'block4_conv1',  'block5_conv1']
        layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',  'block4_conv1']
        style_layer_loss = []
        for layer in layers:
            loss_block = Model(inputs=self.vgg_backbone.input, outputs=self.vgg_backbone.get_layer(layer).output)
            loss_block.trainable = False
            for l in loss_block.layers:
                l.trainable = False
            sum_mean_loss = image_euclidean_loss(
                self.get_mean_and_sigma(K.l2_normalize(loss_block(y_pred), axis=[1, 2]), return_sigma=False, dimension_wise=True),
                self.get_mean_and_sigma(K.l2_normalize(loss_block(s), axis=[1, 2]), return_sigma=False, dimension_wise=True))
            sum_std_loss = image_euclidean_loss(
                self.get_mean_and_sigma(K.l2_normalize(loss_block(y_pred), axis=[1, 2]), return_mean=False, dimension_wise=True),
                self.get_mean_and_sigma(K.l2_normalize(loss_block(s), axis=[1, 2]), return_mean=False, dimension_wise=True))
            style_layer_loss.append(sum_mean_loss + sum_std_loss)
        return K.sum(style_layer_loss) / len(layers)

    @staticmethod
    def get_mean_and_sigma(inputs, return_mean=True, return_sigma=True, dimension_wise=True):
        if return_mean and return_sigma:
            if dimension_wise:
                mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
                sigma = tf.sqrt(tf.add(var, 1e-5))
            else:
                mean, var = tf.nn.moments(inputs, [1, 2, 3])
                sigma = tf.sqrt(tf.add(var, 1e-5))
            return mean, sigma
        if return_mean:
            if dimension_wise:
                mean, _ = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            else:
                mean, var = tf.nn.moments(inputs, [1, 2, 3])
            return mean
        if return_sigma:
            if dimension_wise:
                _, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
                sigma = tf.sqrt(tf.add(var, 1e-5))
            else:
                _, var = tf.nn.moments(inputs, [1, 2, 3])
                sigma = tf.sqrt(tf.add(var, 1e-5))
            return sigma

    def AdaIN(self, inputs):
        f, s = inputs
        meanF, sigmaF = self.get_mean_and_sigma(f)
        meanS, sigmaS = self.get_mean_and_sigma(s)
        return sigmaS * ((f - meanF) / sigmaF) + meanS

    @staticmethod
    def get_decoder():
        decoder_input = Input(shape=(64, 64, 512))
        x = SpatialReflectionPadding()(decoder_input)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D()(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D()(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D()(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
        # x = UpSampling2D()(x)
        decoder_output = Conv2D(3, (3, 3), padding='same', activation='linear')(x)
        decoder = Model(decoder_input, decoder_output, name='decoder')
        # plot_model(decoder, 'decoder.png', show_shapes=True)
        return decoder

    def create_full_model(self):
        style_input = Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), name='style_input')
        content_input = Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), name='content_input')
        style_feats = self.first_feat_extractor(style_input)
        content_feats = self.first_feat_extractor(content_input)
        combined = Lambda(self.AdaIN, name='AdaIN')([content_feats, style_feats])
        # combined = Lambda(lambda x: K.l2_normalize(x, axis=[1, 2]))(combined)
        style_transferred = self.decoder(combined)
        encoder = Model([content_input, style_input], combined)
        full_model = Model([content_input, style_input], style_transferred)
        # plot_model(full_model, 'full_model.png', show_shapes=True)
        return encoder, full_model

    def build_model(self):
        input_style = Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), name='style_input')
        input_content = Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), name='content_input')
        features_combined_t = self.encoder([input_content, input_style])
        style_transferred_t = self.decoder(features_combined_t)
        content_loss_f = image_euclidean_loss(K.l2_normalize(self.first_feat_extractor(style_transferred_t), axis=[1, 2]), K.l2_normalize(features_combined_t, axis=[1, 2]))
        style_loss_f = self.style_loss(style_transferred_t, input_style)
        total_loss = content_loss_f / 20. + 50. * style_loss_f
        update_model = Adam(learning_rate=1e-3).get_updates(total_loss, self.decoder.trainable_weights)
        model_train = K.function([input_content, input_style], [total_loss, content_loss_f, style_loss_f], update_model)
        return model_train

    def train(self):
        losses = []
        norm = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))) * 255.
        for epoch in range(100_000):
            try:
                content, style = self.dl[0]
            except Exception:
                continue
            loss, c_loss, s_loss = self.model_train([content, style])
            losses.append(loss)

            if epoch % 25 == 0:
                output = self.full_model.predict([self.test_contents, self.test_styles])
                output = norm(output)
                content = norm(self.test_contents)
                style = norm(self.test_styles)
                contents = np.hstack([content[i] for i in range(len(self.test_styles))])
                styles = np.hstack([style[i] for i in range(len(self.test_styles))])
                outputs = np.hstack([output[i] for i in range(len(self.test_styles))])
                mix = np.vstack([contents, styles, outputs]).astype('uint8')
                cv2.imwrite(f'./results/images/epoch_{epoch}.jpg', mix)

            if epoch % 100 == 0:
                self.decoder.save_weights(f'./results/weights/epoch_{epoch}.h5')

            plt.figure(figsize=(10, 8))
            plt.plot(losses, alpha=0.5, c='b', label='Total Loss')
            if epoch > 60:
                wl = int(epoch * 0.6)
                plt.plot(savgol_filter(losses, wl if wl % 2 == 1 else wl + 1, 2), c='m', label='Smoothed Loss')
            plt.legend()
            plt.savefig(f'./results/loss.jpg')
            plt.clf()
            plt.close()

            print(f'Epoch: {epoch}\tLoss: [Style: {np.mean(s_loss)}, Content: {np.mean(c_loss)}, Total: {np.mean(loss)}]')


if __name__ == '__main__':
    st = StyleTransfer()
    st.train()
