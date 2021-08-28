from typing import List
from fastapi import FastAPI
from model import EncoderRNN, AttnDecoderRNN, Sentence2Vec
import torch
import numpy as np
from skimage.color import lab2rgb
from pydantic import BaseModel

app = FastAPI()

encoder = EncoderRNN(hidden_size=150, n_layers=1, dropout_p=0)
decoder = AttnDecoderRNN(hidden_size=150, n_layers=1, dropout_p=0)
sen2vec = Sentence2Vec()
encoder.load_state_dict(
    torch.load("./ckpt/ckpt_666.pt", map_location=lambda storage, loc: storage)[
        "encoder"
    ]
)
decoder.load_state_dict(
    torch.load("./ckpt/ckpt_666.pt", map_location=lambda storage, loc: storage)[
        "decoder_state_dict"
    ]
)
encoder.eval()
decoder.eval()


def lab2rgb_1d(in_lab, clip=True):
    tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant="D50").flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    return tmp_rgb


class InputText(BaseModel):
    input_text: str = None


class OutputColor(BaseModel):
    colors: List[str] = []


@app.post("/")
def generate_color(data: InputText, response_model=OutputColor):
    text = sen2vec.embed(data.input_text)["pooler_output"].unsqueeze(0)
    rgb_5 = list()

    batch_size = text.size(0)
    nonzero_indices = list(torch.nonzero(text)[:, 0])
    each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

    palette = torch.FloatTensor(batch_size, 3).zero_()
    fake_palettes = torch.FloatTensor(batch_size, 15).zero_()
    encoder_hidden = encoder.init_hidden(batch_size)

    encoder_outputs, decoder_hidden, _, _ = encoder(text, encoder_hidden)
    decoder_hidden = decoder_hidden.squeeze(0)

    for i in range(5):
        palette, _, decoder_hidden = decoder(
            palette, decoder_hidden, encoder_outputs, each_input_size, i
        )

        fake_palettes[:, 3 * i : 3 * (i + 1)] = palette

    fake_palettes = fake_palettes.squeeze(0)
    for k in range(5):
        lab = np.array(
            [
                fake_palettes.data[3 * k],
                fake_palettes.data[3 * k + 1],
                fake_palettes.data[3 * k + 2],
            ],
            dtype="float64",
        )
        rgb = lab2rgb_1d(lab)
        rgb = rgb * 255
        rgb_5.append('#%02x%02x%02x' % tuple([int(value) for value in rgb]))

    return {"colors": rgb_5}
