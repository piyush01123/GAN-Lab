from PIL import Image, ImageDraw
import os
import glob

names = ['img{:02d}.gif'.format(i) for i in range(20)]
# Create the individual frames as png images
im = Image.new("RGB", (200, 200), 'green')
pos = 0
for n in names:
    frame = im.copy()
    draw = ImageDraw.Draw(frame)
    draw.ellipse((pos, pos, 50+pos, 50+pos),
                 'red')
    frame.save(n)
    pos += 10
# Open all the frames
images = []
for n in names:
    frame = Image.open(n)
    images.append(frame)
# Save the frames as an animated GIF
images[0].save('anicircle.gif',
               save_all=True,
               append_images=images[1:],
               duration=1000,
               loop=0)
for f in glob.glob('img*.gif'):
    os.remove(f)
