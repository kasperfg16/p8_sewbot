import decimal
import numpy as np

descretizetion = 100

# Create a float16 number
number = np.int16(np.pi*descretizetion)
print(number)

number = np.float32(number/descretizetion)
print(number)


print(np.deg2rad([10, 360, -90]))