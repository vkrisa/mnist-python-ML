from PIL import Image

data = "/home/vargak/work/3.png"

im = Image.open(data).convert('L')

im.show()
im.save('3grey.png')
