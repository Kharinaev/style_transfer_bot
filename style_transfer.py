import torchvision.transforms as tt
import torch
from torch.autograd import Variable
import os
import urllib.request
from PIL import Image
from msgnet import MSGNet
from image_preprocess import *


IMAGE_SIZE = 512
NORM_STATS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
if not os.path.isfile('21styles.model'):
    urllib.request.urlretrieve(
        'https://www.dropbox.com/s/2iz8orqqubrfrpo/21styles.model?dl=1',
        '21styles.model'
    )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = torch.load('21styles.model')
model_dict_clone = model_dict.copy()
for key, value in model_dict_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del model_dict[key]
net = MSGNet(ngf=128)
net.load_state_dict(model_dict, False)


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return tt.Pad(padding, 0, 'constant')(image)


def style_transfer(directory):
    # content_pil = Image.open(directory+'/content.jpg')
    # style_pil = Image.open(directory+'/style.jpg')
    # content_size = content_pil.size
    # cur_size = max(content_size)

    # print(f'Image size before stylizing {IMAGE_SIZE}')
    # ratio = IMAGE_SIZE / cur_size
    # new_size = (int(content_size[1] * ratio), int(content_size[0] * ratio))
    # print(f'New size is {new_size}')
    # transforms = tt.Compose([
    #                 tt.Resize(new_size),
    #                 SquarePad(),
    #                 tt.ToTensor(),
    # ])

    # content_img = transforms(content_pil).unsqueeze(0).to(device)
    # style_img = transforms(style_pil).unsqueeze(0).to(device)
    content_img = tensor_load_rgbimage(directory+'/content.jpg', size=IMAGE_SIZE, keep_asp=True).unsqueeze(0)
    style_img = tensor_load_rgbimage(directory + '/style.jpg', size=IMAGE_SIZE, keep_asp=True).unsqueeze(0)
    style_v = Variable(preprocess_batch(style_img))
    content_v = Variable(preprocess_batch(content_img))
    net.setTarget(style_v)
    output = net(content_v)
    # return tt.CenterCrop(new_size)(output[0])
    tensor_save_bgrimage(output.data[0], directory+'/stylized.jpg', False)
