import streamlit as st

import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from models.colorization import ColorizationGenerator
from models.context_embedding import ContextEmbedding
from models.text2palette import T2PGenerator
import tensorflow as tf
import colorsys
import configparser


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
        palette = [0.] * 15
        palette_hex = []

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
        plt.imshow(image)

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)

        plt.close()

        return Image.open(buf), palette_hex


class ColorizationDemo:
    def __init__(self, checkpoint_path):
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

        y = self.ctx([image], [text], [category], np.array([palette]))

        image = tf.image.resize_with_pad(image, 256, 256)
        image = tf.image.rgb_to_grayscale(image)
        image = image / 127.5 - 1
        generated_image = self.gen([np.array([image]), y])

        return generated_image.numpy()[0] * 0.5 + 0.5


def hex_to_rgb(hex):
    '''
    Arguments:
        hex: '#abcdef'
    '''
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:], 16)
    return [r, g, b]


@st.cache(allow_output_mutation=True)
def init():
    config = configparser.ConfigParser()
    config.read('config.ini')
    demo_config = config['streamlit']
    # define models
    t2p_demo = Text2PaletteDemo(
        checkpoint_path=demo_config['text2palette_checkpoint_path'])
    col_demo = ColorizationDemo(
        checkpoint_path=demo_config['colorization_checkpoint_path'])

    return t2p_demo, col_demo


if __name__ == '__main__':
    # initialize model
    t2p_demo, col_demo = init()

    st.title('Automatic Coloring Tool for Chinese Youth Subculture')

    genre = st.sidebar.radio(
        "选择步骤:",
        ('1: Generate Palette', '2: Adjust Palette (not necessary)', '3: Colorization'))

    st.sidebar.subheader('Basic Input')
    f = st.sidebar.file_uploader(
        'upload', type=None, accept_multiple_files=False, key=None)
    if not f:
        st.warning('Upload your file')
        st.stop()

    image = Image.open(f).convert('RGB')
    st.sidebar.image(image, use_column_width=True)

    category = st.sidebar.selectbox(
        'Which category you are design for?',
        ['独立', '电子', 'hiphop', '朋克', '金属', '核', '噪音', '设计', '独立动画', '新媒体艺术', '插画', '摩登天空'])

    if not category:
        st.warning('choose category')
        st.stop()

    st.write('You selected: ', category)

    input_words = st.sidebar.text_input(
        'Input any Chinese words or sentence here:', max_chars=512, type='default')

    if not input_words:
        st.warning('input')
        st.stop()

    if genre == '1: Generate Palette':
        st.subheader('Generated Palette')

        _, palette_hex = t2p_demo.generate(
            np.array(image), input_words, category)

        palette_img = []
        for item in palette_hex:
            palette_img.append(Image.new('RGB', (100, 100), color=item))

        st.image(palette_img)

        st.subheader('Please copy the palette HEX below: ')
        st.write(' '.join(palette_hex))

    if genre == '2: Adjust Palette (not necessary)':
        st.subheader('Adjust Your Palette')

        # 正文
        st.write('Paste your HEX here:')
        words = st.text_input('在此处粘贴您的色板代码：', value='#E9E3D1 #513624 #55442C #947550 #7B623F',
                              max_chars=None, key=None, type='default')
        st.write('You want to change palette: ', words)

        HEX = str.split(words)

        rgb_list = []
        for k in range(0, 5):
            rgb_list.append(hex_to_rgb(HEX[k]))

        palette = rgb_list

        hsl = []

        palette_list = []
        # 建个画板开始画
        for i in range(0, 5):

            r = palette[i][0]
            g = palette[i][1]
            b = palette[i][2]
            HEX = '#%02X%02X%02X' % (r, g, b)
            hsl.append(colorsys.rgb_to_hls(r/255, g/255, b/255))
            palette_list.append(Image.new('RGB', (100, 100), color=HEX))

        st.image(palette_list)

        st.write('The adjustment shows below: ')

        h = []
        l = []
        s = []
        # 更改hsl
        # st.sidebar.write(hsl[color_num][0]*360,hsl[color_num][1],hsl[color_num][2])
        for i in range(0, 5):
            st.sidebar.write("——————Change color: ", i+1, '——————')

            h.append(st.sidebar.slider('Hue:', 0.0, 360.0, hsl[i][0]*360))
            st.sidebar.write("Hue is ", '%.2f' % h[i], '°')

            l.append(st.sidebar.slider('Lightness:', 0.0, 1.0, hsl[i][1]))
            st.sidebar.write("Lightness is ", '{:.2%}'.format(l[i]))

            s.append(st.sidebar.slider('Saturation:', 0.0, 1.0, hsl[i][2]))
            st.sidebar.write("Saturation is ", '{:.2%}'.format(s[i]))

        HEX_change = []

        for i in range(0, 5):
            rgb_change = colorsys.hls_to_rgb(h[i]/360, l[i], s[i])
            r = int(rgb_change[0]*255)
            g = int(rgb_change[1]*255)
            b = int(rgb_change[2]*255)

            HEX_change.append('#%02X%02X%02X' % (r, g, b))

        palette_images = [Image.new('RGB', (100, 100), color=item)
                          for item in HEX_change]

        st.image(palette_images)

        st.subheader(
            'Please copy the palette HEX that you are satisfied with: ')
        st.write(HEX_change[0], HEX_change[1],
                 HEX_change[2], HEX_change[3], HEX_change[4])

    if genre == '3: Colorization':
        st.subheader('Colorization!')

        final_palette_hex = st.text_input('palette')

        if not final_palette_hex:
            st.warning('paste your palette here')
            st.stop()

        palette_img = []
        for item in final_palette_hex.split(' '):
            palette_img.append(Image.new('RGB', (100, 100), color=item))

        st.image(palette_img)

        final_palette_hex_list = final_palette_hex.split(' ')
        final_palette_rgb_list = []
        for item in final_palette_hex_list:
            r = int(item[1:3], 16)
            g = int(item[3:5], 16)
            b = int(item[5:], 16)
            final_palette_rgb_list.append(r)
            final_palette_rgb_list.append(g)
            final_palette_rgb_list.append(b)

        result = col_demo.generate(
            np.array(image), input_words, category, np.array(final_palette_rgb_list) / 255.)

        st.image(result)
