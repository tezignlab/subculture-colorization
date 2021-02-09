from models.colorization import ColorizationGenerator, ColorizationDiscriminator
from models.context_embedding import ContextEmbedding

import tensorflow as tf
from tensorflow.keras.preprocessing import image as tfimage
import numpy as np

import argparse
import configparser
import os

import matplotlib.pyplot as plt
import seaborn as sns


# read config
config = configparser.ConfigParser()
config.read('config.ini')
training_config = config['colorization']

if not os.path.exists(training_config['checkpoint_dir']):
    os.makedirs(training_config['checkpoint_dir'])

if not os.path.exists(training_config['sample_dir']):
    os.makedirs(training_config['sample_dir'])

# create model
gen = ColorizationGenerator()
dis = ColorizationDiscriminator()
ctx = ContextEmbedding()

# define loss
def dis_loss(D_real, D_fake):
    '''
    Discriminator loss is same as the least square GAN (LSGAN)

    Arguments:
        D_real
        D_fake

    '''
    loss_D_real = tf.reduce_mean(tf.nn.l2_loss(D_real - tf.ones_like(D_real)))
    loss_D_fake = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.zeros_like(D_fake)))

    loss_D = loss_D_real + loss_D_fake

    return loss_D

def gen_loss(G_real, G_recon, D_fake):
    '''

    Arguments:
        G_real: ground truth result
        G_recon: generated result
        D_fake: the output of D when input G_recon
    '''
    LAMBDA = 100

    loss_Gls = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.ones_like(D_fake)))

    recon_loss = tf.reduce_sum(tf.pow(G_recon - G_real, 2), 1)
    recon_loss = tf.reduce_mean(recon_loss)

    loss_G = loss_Gls + LAMBDA * recon_loss

    return loss_G

# define optimizer
gen_optimizer = tf.keras.optimizers.SGD(learning_rate=training_config.getfloat('learning_rate'))
dis_optimizer = tf.keras.optimizers.SGD(learning_rate=training_config.getfloat('learning_rate'))

# define checkpoint manager
ckpt = tf.train.Checkpoint(
    gen_optimizer=gen_optimizer,
    dis_optimizer=dis_optimizer,
    gen=gen,
    dis=dis,
    ctx=ctx
)
ckpt_manager = tf.train.CheckpointManager(ckpt, training_config['checkpoint_dir'],
                                          max_to_keep=training_config.getint('checkpoint_max_to_keep'))


def sample(file_name, output_dir):
    dataset = tf.data.experimental.make_csv_dataset('./data/preprocessed_data.csv', batch_size=1)
    dataset = dataset.shuffle(buffer_size=100)

    for idx, raw_data in enumerate(dataset):
        if idx >= 5:
            break

        image = parse_image(raw_data['image'])
        text = parse_text(raw_data['text'])
        category = parse_category(raw_data['category'])
        palette = parse_color(raw_data['colors']).numpy()[0]

        palette_hex = []
        for i in range(5):
            r = int(palette[i * 3 + 0] * 255)
            g = int(palette[i * 3 + 1] * 255)
            b = int(palette[i * 3 + 2] * 255)

            palette_hex.append('#{:02X}{:02X}{:02X}'.format(r, g, b))

        raw_image = []
        gray_image = []
        rgb_image = []
        assert type(image[0]) is str
        for item in image:
            img_item = tfimage.load_img(item)
            img_item = tfimage.img_to_array(img_item)
            raw_image.append(img_item)
            img_item = tf.image.resize_with_pad(img_item, 256, 256)
            img_item = img_item / 127.5 - 1
            rgb_image.append(img_item)

            img_item = tf.image.rgb_to_grayscale(img_item)
            gray_image.append(img_item)

        raw_image = np.stack(raw_image)
        gray_image = np.stack(gray_image)
        rgb_image = np.stack(rgb_image)

        y = ctx(raw_image, text, category, np.array([palette]))
        generated_img = gen([gray_image, y])

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.figure()
        data = np.array(range(5)).reshape((1, 5))

        plt.subplot(1, 4, 1)
        plt.title('input image')
        plt.axis('off')
        plt.imshow((gray_image[0] + 1) * 0.5, cmap='gray')

        plt.subplot(1, 4, 2)
        plt.title('palette')
        color_map = sns.color_palette(palette=palette_hex, as_cmap=True)
        plt.axis('off')
        sns.heatmap(data, cmap=color_map, cbar=False)

        plt.subplot(1, 4, 3)
        plt.title('generated image')
        plt.axis('off')
        plt.imshow((generated_img[0] + 1) * 0.5)

        plt.subplot(1, 4, 4)
        plt.title('real image')
        plt.axis('off')
        plt.imshow((rgb_image[0] + 1) * 0.5)

        plt.savefig(os.path.join(output_dir, file_name) + '-%d.jpg' % idx)
        plt.close()



def train_step(image, text, category, palette):
    raw_image = []
    gray_image = []
    rgb_image = []
    assert type(image[0]) is str
    for item in image:
        img_item = tfimage.load_img(item)
        img_item = tfimage.img_to_array(img_item)
        raw_image.append(img_item)

        target_size = int((1. + np.random.rand()) * 256)
        img_item = tf.image.resize(img_item, [target_size, target_size])
        img_item = tf.image.random_crop(img_item, size=(256, 256, 3))
        img_item = tf.image.random_flip_left_right(img_item)
        img_item = tf.image.random_flip_up_down(img_item)

        img_item = img_item / 127.5 - 1

        rgb_image.append(img_item)

        img_item = tf.image.rgb_to_grayscale(img_item)
        gray_image.append(img_item)

    raw_image = np.stack(raw_image)
    gray_image = np.stack(gray_image)
    rgb_image = np.stack(rgb_image)

    with tf.GradientTape(persistent=True) as tape:
        context = ctx(raw_image, text, category, palette)
        G_recon = gen([gray_image, context], training=True)
        D_fake = dis([G_recon, context], training=True)
        G_loss = gen_loss(rgb_image, G_recon, D_fake)
    
    gen_variables = gen.trainable_variables + ctx.trainable_variables
    gen_gradients = tape.gradient(G_loss, gen_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, gen_variables))

    with tf.GradientTape(persistent=True) as tape:
        context = ctx(raw_image, text, category, palette)
        G_recon = gen([gray_image, context], training=True)
        D_fake = dis([G_recon, context], training=True)
        G_loss = gen_loss(rgb_image, G_recon, D_fake)
    
    gen_variables = gen.trainable_variables + ctx.trainable_variables
    gen_gradients = tape.gradient(G_loss, gen_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, gen_variables))

    with tf.GradientTape(persistent=True) as tape:
        context = ctx(raw_image, text, category, palette)
        G_recon = gen([gray_image, context], training=True)

        D_real = dis([rgb_image, context], training=True)
        D_fake = dis([G_recon, context], training=True)

        G_loss = gen_loss(rgb_image, G_recon, D_fake)
        D_loss = dis_loss(D_real, D_fake)
    
    dis_variables = dis.trainable_variables + ctx.trainable_variables
    gen_variables = gen.trainable_variables + ctx.trainable_variables

    dis_gradients = tape.gradient(D_loss, dis_variables)
    dis_optimizer.apply_gradients(zip(dis_gradients, dis_variables))

    gen_gradients = tape.gradient(G_loss, gen_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, gen_variables))

    return G_loss, D_loss

# some preprocess function for dataset
def parse_image(images):
    final_image_list = []
    for item in images:
        final_image_list.append(os.path.join('./data/images', item.numpy().decode('utf-8')))

    return final_image_list

def parse_text(text):
    final_text_list = []
    for item in text:
        final_text_list.append(item.numpy().decode('utf-8'))

    return final_text_list

def parse_category(category):
    final_category_list = []
    for item in category:
        final_category_list.append(item.numpy().decode('utf-8'))

    return final_category_list

def parse_color(colors):
    '''
    Arguments:
        colors: (batch_size, 15)
    '''
    color_list = tf.strings.split(colors, ',')

    palette = color_list.numpy()
    final_palette = []
    for idx in range(palette.shape[0]):
        final_palette.append([float(c) / 255. for c in palette[idx]])

    return tf.constant(final_palette)

def train():
    dataset = tf.data.experimental.make_csv_dataset('./data/preprocessed_data.csv', batch_size=training_config.getint('batch_size'))
    dataset = dataset.repeat(count=None)
    dataset = dataset.shuffle(buffer_size=100)

    for idx, raw_data in enumerate(dataset):
        if idx >= int(training_config.getfloat('max_iteration_number')):
            break

        image = parse_image(raw_data['image'])
        text = parse_text(raw_data['text'])
        category = parse_category(raw_data['category'])
        palette = parse_color(raw_data['colors'])

        G_loss, D_loss = train_step(image=image, 
                                    text=text,
                                    category=category,
                                    palette=palette)
        
        if (idx + 1) % int(training_config.getfloat('print_every')) == 0:
            print("Iteration: {:5d}, Loss = (G: {:.8f}, D: {:.8f}).".format(
                idx + 1, G_loss, D_loss))

        if (idx + 1) % int(training_config.getfloat('checkpoint_every')) == 0:
            sample(str(idx + 1), training_config['sample_dir'])
            ckpt_manager.save()
            print('Checkpoint %s saved.' % (idx + 1))

def test(ckpt_path, output_dir):
    ckpt.restore(ckpt_path)
    sample('test', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--output_dir', default=None)

    args = parser.parse_args()

    if args.train:
        train()
    if args.test:
        assert args.checkpoint_path and args.output_dir

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        test(args.checkpoint_path, args.output_dir)
