from PIL import Image
import os,sys

mw = 100
ms = 20

msize = mw*ms
toImage = Image.new('RGB',(2000,2000))
fromImage = Image.open(r"C:/Users/39107/Desktop/love/下载.jpg" )
fromImage = fromImage.resize((100,100),Image.ANTIALIAS)
toImage.show()
toImage.save('C:/Users/39107/Desktop/ta.jpg')