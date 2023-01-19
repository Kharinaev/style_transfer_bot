from PIL import Image, ImageDraw, ImageFont
import os


def add_text_to_images(images):
    # path_font = "C:/Users/akhar/Documents/R/win-library/4.1/rmarkdown/rmd/h/bootstrap/css/fonts/Ubuntu.ttf"
    # path_font = "Pillow/Tests/fonts/FreeMono.ttf"
    # font = ImageFont.truetype(path_font, 14)
    for i, image in enumerate(images):
        draw = ImageDraw.Draw(image)
        text = f'{i + 1}/{len(images)}'

        W, H = image.size
        w, h = draw.textsize(text, font=draw.font)
        x_text, y_text = (W - w) / 2, H - h
        facet = 2
        x0, y0 = x_text - facet, y_text - facet
        x1, y1 = x_text + w + facet, y_text + h + facet

        draw.rectangle(((x0, y0), (x1, y1)), fill='white')
        draw.text((x_text, y_text), text, fill="black", font=draw.font)
    return images


def images_to_gif(fname, all_filenames, preffix, dur_per_frame, suffix, add_text=True, sort=False, sort_idx=0):
    if sort:
        all_filenames.sort(key=lambda x: int(x.split('.')[sort_idx]))
    image_fnames = [preffix + name for name in all_filenames]
    if not image_fnames: return
    frames = [Image.open(image) for image in image_fnames]
    if add_text:
        frames = add_text_to_images(frames)
    frame_one = frames[0]
    frame_one.save(f'{fname + suffix}.gif', format="GIF", append_images=frames,
                   save_all=True, duration=dur_per_frame * len(frames), loop=0)


def delete_files(all_filenames, preffix):
    for name in all_filenames:
        os.remove(preffix + name)