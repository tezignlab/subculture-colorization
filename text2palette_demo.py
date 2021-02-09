from models.text2palette import T2PGenerator
from models.context_embedding import ContextEmbedding
import tensorflow as tf
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os


class Text2PaletteDemo:
    def __init__(self, checkpoint_path):
        self.gen = T2PGenerator()
        self.ctx = ContextEmbedding()

        self.ckpt = tf.train.Checkpoint(
            gen=self.gen,
            ctx=self.ctx
        )

        self.ckpt.restore(checkpoint_path)

    def generate(self, image, text, category):
        assert type(image) is str
        image_path = image
        palette = [0.] * 15
        palette_hex = []

        image = tf.keras.preprocessing.image.load_img(image)
        image = tf.keras.preprocessing.image.img_to_array(image)

        for _ in range(5):
            z = np.random.normal(0., 1., size=(1, 128)).astype(np.float32)

            y = self.ctx([image], [text], [category], np.array([palette]))
            new_color = self.gen(z, y)[0]

            r = int(new_color[0] * 255)
            g = int(new_color[1] * 255)
            b = int(new_color[2] * 255)
            palette_hex.append('#{:02X}{:02X}{:02X}'.format(r, g, b))

            palette = np.array(list(palette[3:]) + list(new_color.numpy()))

        color_map = sns.color_palette(palette=palette_hex, as_cmap=True)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title('generated palette')
        data = np.array(range(5)).reshape((1, 5))
        plt.axis('off')
        sns.heatmap(data, cmap=color_map, cbar=False)

        plt.subplot(1, 2, 2)
        plt.title('image')
        plt.axis('off')
        plt.imshow(Image.open(image_path))

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        plt.close()

        return Image.open(buf), palette_hex


if __name__ == '__main__':
    demo = Text2PaletteDemo(checkpoint_path='./ckpt/210129/ckpt-70')
    img_dir = './data/images'
    output_dir = './output-t2p/multi-modal-test'

    with open('./test/category.txt', 'r') as f:
        category = f.readlines()

    with open('./test/image.txt', 'r') as f:
        images = f.readlines()

    with open('./test/text.txt', 'r') as f:
        text = f.readlines()

    for cat in category:
        cat_strip = cat.strip()
        for img in images:
            img_path = img.strip() + '.jpg'
            for txt in text:
                txt_strip = txt.strip()

                for idx in range(5):
                    generated_img, palette_hex = demo.generate(
                        os.path.join(img_dir, img_path), txt_strip, cat_strip)
                    save_path = os.path.join(os.path.join(
                        output_dir, cat_strip), img.strip())

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    generated_img.save(os.path.join(
                        save_path, txt_strip[:5] + ' ' + ''.join(palette_hex) + '.png'))

                    print('%s %s %s - %d' %
                          (cat_strip, img_path, txt_strip, idx))
