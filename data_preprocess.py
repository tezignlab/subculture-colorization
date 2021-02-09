import os
import csv
import requests
from PIL import Image
from io import BytesIO

raw_data_path = './data/url_text_ctg_rgb.csv'  # raw data file path

'''
raw data is a csv file with 4 columns: url, text, category and color.

url: url for raw images
text: text extracted from the page containing the raw image
category: a value from category.txt
color: a string with 15 integers (0-255) joined by comma
'''


def download_image_and_write_csv():
    '''
    read raw data, and download image according to url
    save image in './data/images' with row number as file name
    then replace url with file name and write in a new csv file
    '''
    image_path = './data/images'  # where we save downloaded images

    result_file = open('./data/preprocessed_data.csv', 'w', encoding='utf-8')
    result_writer = csv.writer(result_file)

    error_file = open('./error_download_image_and_write_csv.txt', 'a')

    # some website will forbid direct request of python
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9.1.6) ",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-us",
        "Connection": "keep-alive",
        "Accept-Charset": "GB2312,utf-8;q=0.7,*;q=0.7"
    }
    with open(raw_data_path, 'r') as f:
        data = list(csv.reader(f))

    for idx, item in enumerate(data):
        if idx == 0:
            result_writer.writerow(['image', 'text', 'category', 'colors'])
            continue

        target_file_name = "{:04d}.png".format(idx)
        target_file_path = os.path.join(image_path, target_file_name)

        if os.path.exists(target_file_path):
            continue

        try:
            url = item[0]
            img = Image.open(
                BytesIO(requests.get(url, headers=headers).content))
            # some images are in png format
            # cannot be saved as jpg directly
            img = img.convert('RGB')
            img.save(os.path.join(image_path, "{:04d}.jpg".format(idx)))
            print('%d downloaded' % idx)

            result_writer.writerow(
                [target_file_name, item[1], item[2], item[3]])
        except Exception as e:
            error_file.write(item + '\n')
            print(idx, e)


def augment_preprocess_data():
    '''
    augment data by adding remove some tail palette,

    so the generator can generate next color best matched with 'context' palette
    ---
    e.g.
    raw palette: 
    [a, b, c, d, e]

    augment palette:
    [a, b, c, d, e]
    [0, a, b, c, d]
    [0, 0, a, b, c]
    [0, 0, 0, a, b]
    [0, 0, 0, 0, a]
    '''
    raw_data = './data/preprocessed_data.csv'
    result_data = './data/augment_data.csv'

    raw_reader = csv.reader(open(raw_data, 'r'))
    result_writer = csv.writer(open(result_data, 'w'))

    for idx, item in enumerate(raw_reader):
        if idx == 0:
            result_writer.writerow(item)
            continue

        temp_color = item[3]
        for i in range(5):
            colors = ['0'] * (i * 3)

            colors += temp_color.split(',')

            colors = colors[:15]

            result_writer.writerow(
                [item[0], item[1], item[2], ','.join(colors)])
