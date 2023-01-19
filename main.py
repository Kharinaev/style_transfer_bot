import os.path
import shutil
from telegram.ext import Updater, CommandHandler, MessageHandler, TypeHandler
from bot import *


def prepare_dirs():
    if os.path.exists('chats'):
        shutil.rmtree('chats')
    os.mkdir('chats')


def main() -> None:
    prepare_dirs()
    with open('token.txt') as f:
        token = f.read().strip()
    updater = Updater(token)

    updater.dispatcher.add_handler(CommandHandler("start", start_command))
    updater.dispatcher.add_handler(CommandHandler("help", help_command))
    updater.dispatcher.add_handler(CommandHandler("gif", gif_command))
    # updater.dispatcher.add_handler(CommandHandler("size128", partial(set_image_size, size=128)))
    # updater.dispatcher.add_handler(CommandHandler("size256", partial(set_image_size, size=256)))
    # updater.dispatcher.add_handler(CommandHandler("size512", partial(set_image_size, size=512)))
    updater.dispatcher.add_handler(MessageHandler(
        (
            (check_caption('style') | check_caption('content')) &
            Filters.photo
        ),
        image_handler
    ))
    updater.dispatcher.add_handler(TypeHandler(Update, unknown_message_format))

    updater.start_polling()
    print('Bot started')
    updater.idle()


if __name__ == '__main__':
    main()