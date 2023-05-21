import decimal
import numpy as np

# Create a float16 number
number = np.int16(1.11*100)
print(number)

number = np.float32(number/100)
print(number)