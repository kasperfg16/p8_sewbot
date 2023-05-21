import decimal
import numpy as np

descretizetion = 0.5

# Create a float16 number
number = np.int16(153.0*descretizetion)
print(number)

number = np.float32(number/descretizetion)
print(number)


print(np.deg2rad([10, 360, -90]))