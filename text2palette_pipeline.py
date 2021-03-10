from models.text2palette import T2PGenerator, T2PDiscriminator
from models.context_embedding import ContextEmbedding

import tensorflow as tf
import numpy as np

import argparse
import configparser
import os

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import datetime

# read config
config = configparser.ConfigParser()
config.read('config.ini')
training_config = config['text2palette']

if not os.path.exists(training_config['checkpoint_dir']):
    os.makedirs(training_config['checkpoint_dir'])

if not os.path.exists(training_config['sample_dir']):
    os.makedirs(training_config['sample_dir'])

log_dir = os.path.join(training_config['log_dir'], datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

if not os.path.exists(log_dir):
    os.makedirs(training_config['log_dir'])

# create model
gen = T2PGenerator()
dis = T2PDiscriminator()
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
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=training_config.getfloat('learning_rate'),
                                         beta_1=training_config.getfloat('beta_1'))
dis_optimizer = tf.keras.optimizers.Adam(learning_rate=training_config.getfloat('learning_rate'),
                                         beta_1=training_config.getfloat('beta_1'))

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
        real_context_palette, real_target_palette = parse_color(raw_data['colors'])
        real_palette = list(real_context_palette.numpy()[0]) + list(real_target_palette.numpy()[0])

        context_palette = [0.] * 15

        raw_image = []

        for item in image:
            img_item = tf.keras.preprocessing.image.load_img(item)
            img_item = img_item.convert('RGB')
            img_item = tf.keras.preprocessing.image.img_to_array(img_item)
            img_item = tf.image.resize(img_item, [224, 224])
            raw_image.append(img_item)
        
        raw_image = np.stack(raw_image)

        for i in range(5):
            z = np.random.normal(0., 1., size=(
                1, training_config.getint('z_dim'))).astype(np.float32)

            y = ctx(raw_image, text, category, np.array([context_palette]))
            new_color = gen(z, y)[0]

            context_palette = np.array(list(context_palette[3:]) + list(new_color.numpy()))
        
        palette_hex = []
        real_palette_hex = []
        for i in range(5):
            r = int(context_palette[i * 3 + 0] * 255)
            g = int(context_palette[i * 3 + 1] * 255)
            b = int(context_palette[i * 3 + 2] * 255)

            palette_hex.append('#{:02X}{:02X}{:02X}'.format(r, g, b))

            r = int(real_palette[i * 3 + 0] * 255)
            g = int(real_palette[i * 3 + 1] * 255)
            b = int(real_palette[i * 3 + 2] * 255)

            real_palette_hex.append('#{:02X}{:02X}{:02X}'.format(r, g, b))
        
    
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.figure()
        data = np.array(range(5)).reshape((1, 5))

        plt.subplot(2, 2, 1)
        plt.title('generated palette')
        color_map = sns.color_palette(palette=palette_hex, as_cmap=True)
        plt.axis('off')
        sns.heatmap(data, cmap=color_map, cbar=False)

        plt.subplot(2, 2, 3)
        plt.title('real palette')
        color_map = sns.color_palette(palette=real_palette_hex, as_cmap=True)
        plt.axis('off')
        sns.heatmap(data, cmap=color_map, cbar=False)

        plt.subplot(1, 2, 2)
        plt.title('text: %s; category: %s' % (text[0], category[0]))
        plt.axis('off')
        plt.imshow(Image.open(image[0]))

        plt.savefig(os.path.join(output_dir, file_name) + '-%d.jpg' % idx)
        plt.close()


def train_step(z, image, text, category, context_palette, target_palette):
    '''

    Arguments:
        z:
        image:
        text:
        category:
        palette:
    '''
    raw_image = []

    for item in image:
        img_item = tf.keras.preprocessing.image.load_img(item)
        img_item = img_item.convert('RGB')
        img_item = tf.keras.preprocessing.image.img_to_array(img_item)
        img_item = tf.image.resize(img_item, [224, 224])
        raw_image.append(img_item)
    
    raw_image = np.stack(raw_image)

    with tf.GradientTape(persistent=True) as tape:
        context = ctx(raw_image, text, category, context_palette)
        G_recon = gen(z, context)

        D_real = dis(target_palette, context)
        D_fake = dis(G_recon, context)

        G_loss = gen_loss(target_palette, G_recon, D_fake)
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

    context_palette = color_list[:, :12].numpy()
    target_palette = color_list[:, 12:].numpy()

    final_context_palette = []
    final_target_palette = []
    for idx in range(context_palette.shape[0]):
        final_context_palette.append([float(c) / 255. for c in context_palette[idx]])
        final_context_palette[-1] = [0., 0., 0.] + final_context_palette[-1]
        final_target_palette.append([float(c) / 255. for c in target_palette[idx]])

    return tf.constant(final_context_palette), tf.constant(final_target_palette)


def train():
    dataset = tf.data.experimental.make_csv_dataset('./data/augment_data.csv', batch_size=training_config.getint('batch_size'))
    dataset = dataset.repeat(count=None)
    dataset = dataset.shuffle(buffer_size=100)

    train_summary_writer = tf.summary.create_file_writer(log_dir)

    for idx, raw_data in enumerate(dataset):
        if idx >= int(training_config.getfloat('max_iteration_number')):
            break

        image = parse_image(raw_data['image'])
        text = parse_text(raw_data['text'])
        category = parse_category(raw_data['category'])
        context_palette, target_palette = parse_color(raw_data['colors'])

        z = np.random.normal(0., 1., size=(
            training_config.getint('batch_size'), training_config.getint('z_dim'))).astype(np.float32)

        G_loss, D_loss = train_step(z=z,
                                    image=image,
                                    text=text,
                                    category=category,
                                    context_palette=context_palette,
                                    target_palette=target_palette
                                    )
        if (idx + 1) % int(training_config.getfloat('print_every')) == 0:
            print("Iteration: {:5d}, Loss = (G: {:.8f}, D: {:.8f}).".format(
                idx + 1, G_loss, D_loss))
                
            with train_summary_writer.as_default():
                tf.summary.scalar('G_loss', G_loss, step=idx)
                tf.summary.scalar('D_loss', D_loss, step=idx)

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
