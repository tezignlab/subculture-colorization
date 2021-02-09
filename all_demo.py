from tensorflow.python.eager.context import Context
from models.text2palette import T2PGenerator
from models.colorization import ColorizationGenerator
from models.context_embedding import ContextEmbedding
import tensorflow as tf
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import csv


class AllDemo:
    def __init__(self, t2p_checkpoint_path, color_checkpoint_path):
        self.t2p = T2PGenerator()
        self.t2p_ctx = ContextEmbedding()
        self.t2p_ckpt = tf.train.Checkpoint(
            gen=self.t2p,
            ctx=self.t2p_ctx
        )
        self.t2p_ckpt.restore(t2p_checkpoint_path)

        self.color_gen = ColorizationGenerator()
        self.color_ctx = ContextEmbedding()
        self.color_ckpt = tf.train.Checkpoint(
            gen=self.color_gen,
            ctx=self.color_ctx
        )
        self.color_ckpt.restore(color_checkpoint_path)

    def generate(self, image, text, category):
        assert type(image) is str

        image_path = image

        palette = [0.] * 15
        palette_hex = []

        image = tf.keras.preprocessing.image.load_img(image)
        image = tf.keras.preprocessing.image.img_to_array(image)

        for _ in range(5):
            z = np.random.normal(0., 1., size=(1, 128)).astype(np.float32)

            y = self.t2p_ctx([image], [text], [category], np.array([palette]))
            new_color = self.t2p(z, y)[0]

            r = int(new_color[0] * 255)
            g = int(new_color[1] * 255)
            b = int(new_color[2] * 255)
            palette_hex.append('#{:02X}{:02X}{:02X}'.format(r, g, b))

            for _ in range(3):
                palette = np.array(list(palette[3:]) + list(new_color.numpy()))

        color_map = sns.color_palette(palette=palette_hex, as_cmap=True)

        y = self.color_ctx([image], [text], [category], np.array([palette]))

        image = tf.image.resize_with_pad(image, 256, 256)
        image = tf.image.rgb_to_grayscale(image)
        image = image / 127.5 - 1
        generated_image = self.color_gen([np.array([image]), y])

        plt.figure()

        plt.subplot(1, 4, 1)
        plt.title('input image')
        plt.axis('off')
        plt.imshow(Image.open(image_path).convert('L'), cmap='gray')

        plt.subplot(1, 4, 2)
        plt.title('generated palette')
        plt.axis('off')
        data = np.array(range(5)).reshape((1, 5))
        sns.heatmap(data, cmap=color_map, cbar=False)

        plt.subplot(1, 4, 3)
        plt.title('colored image')
        plt.axis('off')
        plt.imshow(generated_image.numpy()[0] * 0.5 + 0.5)

        plt.subplot(1, 4, 4)
        plt.title('ground truth')
        plt.axis('off')
        plt.imshow(Image.open(image_path))

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        plt.close()

        return Image.open(buf), palette_hex


if __name__ == '__main__':
    demo = AllDemo(t2p_checkpoint_path='./ckpt-t2p/210129/ckpt-70',
                   color_checkpoint_path='./ckpt-color/210129-3/ckpt-100')

    img_dir = './data/images'
    output_dir = './output/whole-pipeline-test'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_data = './data/preprocessed_data.csv'
    raw_reader = csv.reader(open(raw_data, 'r'))

    for idx, item in enumerate(raw_reader):
        if idx == 0:
            continue

        image_path = item[0].strip()
        text = item[1].strip()
        category = item[2].strip()

        for generated_idx in range(5):
            generated_img, palette_hex = demo.generate(
                os.path.join(img_dir, image_path), text, category)

            generated_img.save(os.path.join(
                output_dir, '%d-%d-%s.jpg' % (idx, generated_idx, ''.join(palette_hex))))

            print('%d %d' % (idx, generated_idx))
