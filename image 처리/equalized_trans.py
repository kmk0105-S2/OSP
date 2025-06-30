import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
pdf = np.loadtxt("PDF.txt", delimiter="\t", usecols=1)
eq_pdf = np.loadtxt("equalized_PDF_gray.txt", delimiter="\t", usecols=1)
trans_func = np.loadtxt("trans_func_eq.txt")
x = np.arange(256)
y_trans = trans_func[:, 1]


fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Original PDF
axs[0].plot(x, pdf, linewidth=1)
axs[0].set_title("Original PDF")
axs[0].set_xticks(range(0, 256, 15))
axs[0].grid(True, axis='y')

# Equalized PDF
axs[1].plot(x, eq_pdf, linewidth=1)
axs[1].set_title("Equalized PDF gray version")
axs[1].set_xticks(range(0, 256, 15))
axs[1].grid(True, axis='y')

# Equalized Function
axs[2].plot(x, y_trans, linewidth=2, color='limegreen')
axs[2].set_title("Equalized Function")
axs[2].set_xlabel("Original value")
axs[2].set_ylabel("Equalized value")
axs[2].grid(True)


plt.tight_layout()
plt.show()