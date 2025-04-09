import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
pdf = np.loadtxt("PDF.txt", delimiter="\t", usecols=1)
mat_pdf = np.loadtxt("matching_PDF_YUV.txt", delimiter="\t", usecols=1)
x = np.arange(256)


fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Original PDF
axs[0].plot(x, pdf, linewidth=1)
axs[0].set_title("Original PDF")
axs[0].set_xticks(range(0, 256, 15))
axs[0].grid(True, axis='y')

# Stretched PDF
axs[1].plot(x, mat_pdf, linewidth=1)
axs[1].set_title("Matching RGB PDF")
axs[1].set_xticks(range(0, 256, 15))
axs[1].grid(True, axis='y')

plt.show()