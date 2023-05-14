import numpy as np

scale = 5

spacing = 0.05
size=np.array([0.015, 0.01])

spacing=spacing/scale
size=size/scale

print('spacing="',spacing,'"')

print('size="',size[0],' ',size[1],'"')