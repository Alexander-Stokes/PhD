import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEL = np.linspace(3.3, 10, 1000)
EDC = np.linspace(5, 100, 1000)
NHS = np.linspace(5, 50, 1000)

hydrogel = np.empty((1000, 6))

# Generate random combinations for each row
for i in range(1000):
    a = np.random.choice(GEL)
    hydrogel[i, 0] = a
    b = np.random.choice(EDC)
    hydrogel[i, 1] = b
    c = np.random.choice(NHS)
    hydrogel[i, 2] = c
    d = a * b * c
    hydrogel[i, 3] = d
    e = a + b + c
    hydrogel[i, 4] = e
    f = e / 3
    hydrogel[i, 5] = f



fakeproto = pd.DataFrame(hydrogel, columns=['GEL', 'EDC', 'NHS', 'd', 'e', 'f'])
fakeproto.to_csv('fakeProtoOutOfRangeInteract.csv', index=False)
