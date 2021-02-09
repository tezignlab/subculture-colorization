from models.colorization import ColorizationGenerator
from models.context_embedding import ContextEmbedding
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns


class ColorizationDemo:
    def __init__(self, checkpoint_path, output_dir):
        self.output_dir = output_dir

        self.gen = ColorizationGenerator()
        self.ctx = ContextEmbedding()

        self.ckpt = tf.train.Checkpoint(
            gen=self.gen,
            ctx=self.ctx
        )

        self.ckpt.restore(checkpoint_path)

    def generate(self, image, text, category, palette):
        '''
        '''

        assert type(image) is str

        image = tf.keras.preprocessing.image.load_img(image)
        image = tf.keras.preprocessing.image.img_to_array(image)
        
        y = self.ctx([image], [text], [category], np.array([palette]))

        image = tf.image.resize_with_pad(image, 256, 256)
        image = tf.image.rgb_to_grayscale(image)
        image = image / 127.5 - 1
        generated_image = self.gen([np.array([image]), y])

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('input image')
        plt.axis('off')
        plt.imshow(image * 0.5 + 0.5, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title('input palette')
        palette_hex = []
        for i in range(5):
            r = palette[3 * i + 0]
            g = palette[3 * i + 1]
            b = palette[3 * i + 2]
            palette_hex.append('#{:02X}{:02X}{:02X}'.format(r, g, b))
        plt.axis('off')
        color_map = sns.color_palette(palette=palette_hex, as_cmap=True)
        data = np.array(range(5)).reshape((1, 5))
        sns.heatmap(data, cmap=color_map, cbar=False)

        plt.subplot(1, 3, 3)
        plt.title('generated image')
        plt.axis('off')
        plt.imshow(generated_image.numpy()[0] * 0.5 + 0.5)
        
        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        plt.close()

        return Image.open(buf)


if __name__ == '__main__':
    demo = ColorizationDemo('./ckpt-color/210129-3/ckpt-80', './output-color/210129-3/test')

    img = demo.generate('./data/images/0101.jpg', '快乐', '电子', [64,125,251,245,70,53,85,138,251,222,231,252,193,85,108])
    img.save(os.path.join(demo.output_dir, 'test-1.png'))

    img = demo.generate('./data/images/0101.jpg', '快乐', '电子', [252,161,28,40,25,4,146,93,18,104,67,10,210,136,21])
    img.save(os.path.join(demo.output_dir, 'test-2.png'))
