import decimal
import numpy as np

# Create a float16 number
number = np.int16(-np.deg2rad(160)*1000)

number = np.float32(number/1000)

print(number)

print(-np.deg2rad(160))

number = np.float32(1989)
print(number)