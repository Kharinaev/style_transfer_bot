# Style transfer bot
Telegram bot that allows you to transfer style from one image to another

## Description

Current bot written in `python-telegram-bot` framework. It uses [PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) implementation of [Neural Style Tranferring Algorithm](https://arxiv.org/abs/1508.06576) developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.


Bot can recieve two images (style and content) and perform style transferring from one to another.   
Example:  
![Alt text](style_transfer_example.jpg?raw=true "Title")

It also can show you a process of stylizing with gif.  
Example:  
![Alt text](process.gif?raw=true)



## Setup
```
git clone https://github.com/Kharinaev/style_transfer_bot.git
pip install python-telegram_bot==13.13 -q -U
echo 'YOUR_BOT_TOKEN' > token.txt
```
```
python style_transfer_bot/main.py
```
