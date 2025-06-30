import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
pdf = np.loadtxt("PDF.txt", delimiter="\t", usecols=1)
st_pdf = np.loadtxt("stretched_PDF.txt", delimiter="\t", usecols=1)
trans_func = np.loadtxt("trans_func_stretch.txt")
x = np.arange(256)
y_trans = trans_func[:, 1]


fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Original PDF
axs[0].plot(x, pdf, linewidth=1)
axs[0].set_title("Original PDF")
axs[0].set_xticks(range(0, 256, 15))
axs[0].grid(True, axis='y')

# Stretched PDF
axs[1].plot(x, st_pdf, linewidth=1)
axs[1].set_title("Stretched PDF")
axs[1].set_xticks(range(0, 256, 15))
axs[1].grid(True, axis='y')

# Stretching Function
axs[2].plot(x, y_trans, linewidth=2, color='royalblue')
axs[2].set_title("Stretching Function")
axs[2].set_xlabel("Original value")
axs[2].set_ylabel("Stretched value")
axs[2].grid(True)


plt.tight_layout()
plt.show()
