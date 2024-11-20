import random
import math

def inverse_transform():
    u = random.uniform(0, 1)
    if u < 0.25:  # First uniform part (1-2 minutes)
        x = u * 4 + 1  # Maps [0, 0.25] to [1, 2]
    elif u < 0.75:  # Second uniform part (3-4 minutes)
        x = (u - 0.25) * 2 + 3  # Maps [0.25, 0.75] to [3, 4]
    else:  # Triangular part (4-6 minutes)
        if u < 0.875:  # Rising part of triangle
            x = 4 + 2 * math.sqrt((u - 0.75) * 2)  # Maps [0.75, 0.875] to [4, 5]
        else:  # Falling part of triangle
            x = 6 - 2 * math.sqrt((1 - u) * 2)  # Maps [0.875, 1] to [5, 6]
    return x


def f(x):
    if 1 <= x < 2:
        return 0.25  # First uniform part
    elif 3 <= x < 4:
        return 0.5   # Second uniform part
    elif 4 <= x < 5:
        return 0.25 * (x - 4)  # Rising part of triangle
    elif 5 <= x < 6:
        return 0.25 * (6 - x)  # Falling part of triangle
    else:
        return 0
    
def rejection_sample():
    while True:
        y = random.uniform(1, 6)  # Generate from covering distribution
        u = random.uniform(0, 1)
        if u <= f(y) / 0.5:  # 0.5 is the maximum of f(x)
            return y
        

def composition():
    u = random.uniform(0, 1)
    if u < 0.25:  
        # First uniform part (1-2 minutes)
        x = random.uniform(1, 2)
    elif u < 0.75:  
        # Second uniform part (3-4 minutes)
        x = random.uniform(3, 4)
    else:  
        # Triangular part (4-6 minutes)
        v = random.uniform(0, 1)
        if v < 0.5:  # Rising part
            x = 4 + math.sqrt(2 * v)  # Maps [0, 0.5] to [4, 5]
        else:  # Falling part
            x = 6 - math.sqrt(2 * (1 - v))  # Maps [0.5, 1] to [5, 6]
    return x

