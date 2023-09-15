#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=np.linspace(40, 100, 7), edgecolor='k')
plt.ylim(0, 30)
plt.xlim(0, 100)
plt.xticks(np.linspace(0, 100, 11))
plt.ylabel('Number of Students')
plt.xlabel('Grades')
plt.title('Project A')
plt.show()
