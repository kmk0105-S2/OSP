import matplotlib.pyplot as plt
import numpy as np

pdf = np.loadtxt("PDF.txt", delimiter="\t", usecols=1)
cdf = np.loadtxt("CDF.txt", delimiter="\t", usecols=1)

x = np.arange(256)

#pdf 그리기
plt.figure(figsize=(10,5))
plt.plot(x, pdf, linewidth=1)
plt.xticks(range(0, 256, 15))
plt.grid(True, axis='y')
plt.title("PDF")
plt.show()

#cdf 그리기
plt.figure(figsize=(10, 5))
plt.plot(x, cdf, linewidth=1)
plt.xticks(range(0,256,15))
plt.grid(True, axis='y')
plt.title("CDF")
plt.show()