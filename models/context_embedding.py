import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np
from transformers import BertTokenizer, TFBertModel


class VisualFeatureExtract:
    def __init__(self):
        self.vgg16_base = tf.keras.applications.VGG16(include_top=False,
                                                      input_shape=(224, 224, 3))
        self.vgg16_feature_extract = tf.keras.Model(
            inputs=self.vgg16_base.input,
            outputs=self.vgg16_base.get_layer('block5_conv3').output)

    def extract(self, img_list, target_size=(224, 224)):
        """Extract feature from image list using VGG16.

        Args:
            img_list: List of numpy array

        Returns:
            numpy array: The feature of the input image list. Size is (list_size, 14, 14, 512).
        """
        new_img_list = []
        for img_item in img_list:
            img_item = tf.image.resize(img_item, [224, 224])
            img_item = tf.image.rgb_to_grayscale(img_item)
            img_item = tf.concat([img_item] * 3, 2)
            new_img_list.append(img_item)

        # pack the images list into a "batch"
        new_img_list = np.stack(new_img_list)

        # it's important to preprocess
        new_img_list = tf.keras.applications.vgg16.preprocess_input(
            new_img_list)

        result = self.vgg16_feature_extract(new_img_list)

        return result


class TextFeatureExtract:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            'uer/chinese_roberta_L-8_H-512')
        self.model = TFBertModel.from_pretrained(
            "uer/chinese_roberta_L-8_H-512")

    def extract(self, sentence_list):
        '''

        Returns:
            output: (list_size, 512)
        '''
        encoded_input = self.tokenizer(
            sentence_list, return_tensors='tf', padding=True)
        output = self.model(**encoded_input)['pooler_output']

        return output


class CategoryAttributeHandler:
    def __init__(self, category_path='./data/category.txt'):
        self.category_list = []
        with open(category_path, 'r') as f:
            for line in f.readlines():
                self.category_list.append(line.strip())

        self.category_idx_dict = {}

        for idx, item in enumerate(self.category_list):
            self.category_idx_dict[item] = idx

    def get(self, category):
        '''

        Returns:
            result: (category size, 10 * category number)
        '''
        indices = []

        for item in category:
            if item not in self.category_list:
                indices.append(-1)
            else:
                indices.append(self.category_idx_dict[item])

        cat_embedding = tf.one_hot(indices, depth=len(self.category_list))

        # repeat for 10 times
        # to increase significance
        result = tf.concat([cat_embedding] * 10, 1)

        return result


class ContextEmbedding(keras.Model):
    def __init__(self):
        super(ContextEmbedding, self).__init__()

        self.img_feature = VisualFeatureExtract()
        self.txt_feature = TextFeatureExtract()
        self.cat_feature = CategoryAttributeHandler()

        activation = tf.nn.relu

        self.img_fc1 = Dense(512, activation=activation)
        self.img_fc2 = Dense(256, activation=activation)
        self.img_fc3 = Dense(128, activation=activation)

        self.txt_fc1 = Dense(256, activation=activation)
        self.txt_fc2 = Dense(256, activation=activation)
        self.txt_fc3 = Dense(128, activation=activation)

        self.cat_fc1 = Dense(128, activation=activation)
        self.cat_fc2 = Dense(128, activation=activation)

        # TODO: 
        self.lambda_img = 0.6
        self.lambda_txt = 0.3
        self.lambda_cat = 0.1

        self.fusion_fc1 = Dense(128, activation=activation)

    def call(self, image, text, category, context_palette):
        '''

        Arguments:
            context_palette: size - (batch, 15)
        '''
        img_fea = self.img_feature.extract(image)   # (batch, 14, 14, 512)
        txt_fea = self.txt_feature.extract(text)    # (batch, 512)
        cat_fea = self.cat_feature.get(category)    # (batch, 10 * category_number)

        img_fea = tf.reduce_mean(img_fea, [1, 2])   # (batch, 512)
        img_fea = self.img_fc1(img_fea)
        img_fea = self.img_fc2(img_fea)
        img_fea = self.img_fc3(img_fea)             # (batch, 128)

        txt_fea = self.txt_fc1(txt_fea)
        txt_fea = self.txt_fc2(txt_fea)
        txt_fea = self.txt_fc3(txt_fea)             # (batch, 128)

        cat_fea = self.cat_fc1(cat_fea)
        cat_fea = self.cat_fc2(cat_fea)             # (batch, 128)

        ctx_fea = self.lambda_img * img_fea + \
            self.lambda_txt * txt_fea + \
            self.lambda_cat * cat_fea               # (batch, 128)

        y = self.fusion_fc1(ctx_fea)                # (batch, 128)

        y = tf.concat([y, context_palette], 1)      # (batch, 128 + 12)

        return y
