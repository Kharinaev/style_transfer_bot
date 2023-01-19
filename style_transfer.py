import torch.optim as optim
import torchvision.transforms as tt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import shutil
import os
from tqdm import tqdm, trange


IMAGE_SIZE = 128
NORM_STATS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = models.vgg19(pretrained=True).features.to(device).eval()


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1, size=[512,512],
                       process_dir=None, log_steps=50):
    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    if process_dir is not None:
        with torch.no_grad():
            tmp = input_img.detach().clamp(0, 1)
            tmp_img = tt.CenterCrop(size)(tmp[0])
            pil_stylized_img = tt.ToPILImage()(tmp_img)
            pil_stylized_img.save(process_dir + f'{0}.jpg')

    optimizer = get_input_optimizer(input_img)

    # print('Optimizing..')
    for i in trange(num_steps):

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            return style_score + content_score

        optimizer.step(closure)

        if i % log_steps == 0:
            if process_dir is not None:
                with torch.no_grad():
                    tmp = input_img.detach().clamp(0, 1)
                    tmp_img = tt.CenterCrop(size)(tmp[0])
                    pil_stylized_img = tt.ToPILImage()(tmp_img)
                    pil_stylized_img.save(process_dir + f'{i+1}.jpg')

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return tt.Pad(padding, 0, 'constant')(image)


def style_transfer(directory, content, style, process_dir=None, num_steps=20, log_steps=2):
    if process_dir is not None:
        if os.path.exists(process_dir):
            shutil.rmtree(process_dir)
        os.mkdir(process_dir)
    content_pil = Image.open(directory+content)
    style_pil = Image.open(directory+style)
    content_size = content_pil.size
    cur_size = max(content_size)

    #
    # print(f'Image size before stylizing {IMAGE_SIZE}')
    ratio = IMAGE_SIZE / cur_size
    new_size = (int(content_size[1] * ratio), int(content_size[0] * ratio))
    # print(f'New size is {new_size}')
    transforms = tt.Compose([
                    tt.Resize(new_size),
                    SquarePad(),
                    tt.ToTensor(),
    ])

    content_img = transforms(content_pil).unsqueeze(0).to(device)
    style_img = transforms(style_pil).unsqueeze(0).to(device)
    input_img = content_img.clone()
    output = run_style_transfer(
        cnn,
        NORM_STATS[0],
        NORM_STATS[1],
        content_img,
        style_img,
        input_img,
        size=new_size,
        process_dir=process_dir,
        log_steps=log_steps,
        num_steps=num_steps
    )
    return tt.CenterCrop(new_size)(output[0])
