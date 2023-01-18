import re
import os.path
import os
from telegram import Update
from telegram.ext import CallbackContext, Filters, BaseFilter
import torchvision.transforms as tt
from style_transfer import style_transfer


def start_command(update: Update, context: CallbackContext):
    name = update.message.chat.first_name
    update.message.reply_text('Hello, ' + name + '!')
    update.message.reply_text('Please share your images separately with captions "style" and "content"')


def help_command(update: Update, context: CallbackContext):
    help_text = ' Bot produces photo style transferring from one image to another'\
                '\n\nShare two images separately with captions "style" and "content"'\
                '\n\nBot will transfer style from "style" image to "content" image'\
                '\n\nFor better quality "style" image should have some special style, '\
                'like Claude Monet or Vincent Van Gogh paintings, or special texture'
    update.message.reply_text(help_text)


def image_handler(update: Update, context: CallbackContext):
    file = update.message.photo[-1].file_id

    chat_id = update.message.chat.id
    directory = f'chats/{chat_id}'
    if not os.path.exists(directory):
        os.mkdir(directory)

    message = update.message.caption.lower()
    obj = context.bot.get_file(file)
    obj.download(directory + f'/{message}.jpg')
    update.message.reply_text(f"Image {message} received")
    print(f"Image {message} received")

    if os.path.isfile(directory + '/style.jpg') & os.path.isfile(directory + '/content.jpg'):
        update.message.reply_text(f"Stylizing...")
        style_transfer(directory)
        # stylized_img = style_transfer(directory)
        # pil_stylized_img = tt.ToPILImage()(stylized_img)
        # pil_stylized_img.save(directory + '/stylized.jpg')
        context.bot.send_photo(
            update.message.chat.id,
            photo=open(directory + '/stylized.jpg', 'rb')
        )
        os.remove(directory + '/content.jpg')
        os.remove(directory + '/style.jpg')


# def set_image_size(update: Update, context: CallbackContext, size=512):
#     update.message.reply_text(f"Image size set to {size}")
#     print(f"Image size set to {size}")
#     IMAGE_SIZE = size
#     print(f'Image size after changing {IMAGE_SIZE}')


def check_caption(s) -> BaseFilter:
    return Filters.caption_regex(re.compile(f'^{s}$', re.IGNORECASE)) & Filters.caption


def unknown_message_format(update: Update, context: CallbackContext):
    update.message.reply_text("Unknown message format, please check /help to see what format bot supports")


