#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image

im = Image.open('4.png')
im.thumbnail((28, 28))
tt =  im.convert('1')
tt.save('test4.bmp')