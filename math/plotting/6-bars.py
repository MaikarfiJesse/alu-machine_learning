#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fruit_bottoms = np.append(np.zeros((1, 3)), fruit.cumsum(axis=0), axis=0)
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
fruit_colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

for i in range(4):
    plt.bar(
        range(3),
        height=fruit[i],
        bottom=fruit_bottoms[i],
        width=0.5,
        color=fruit_colors[i],
        label=fruit_names[i]
    )

plt.yticks(np.linspace(0, 80, 9))
plt.xticks(range(3), ['Farrah', 'Fred', 'Felicia'])
plt.ylabel('Quantity of Fruit')
plt.legend(loc='upper right')
plt.title('Number of Fruit per Person')
plt.show()
