import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
pdf_R = np.loadtxt("original_R_YUV.txt", delimiter="\t", usecols=1)
pdf_G = np.loadtxt("original_G_YUV.txt", delimiter="\t", usecols=1)
pdf_B = np.loadtxt("original_B_YUV.txt", delimiter="\t", usecols=1)

pdf_RO = np.loadtxt("output_R_YUV.txt", delimiter="\t", usecols=1)
pdf_GO = np.loadtxt("output_G_YUV.txt", delimiter="\t", usecols=1)
pdf_BO = np.loadtxt("output_B_YUV.txt", delimiter="\t", usecols=1)

x = np.arange(256)  # 밝기 값 0~255

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Original R PDF
axs[0].plot(x, pdf_R, color='r', linewidth=1)
axs[0].set_title("Original R")
axs[0].set_xticks(range(0, 256, 15))
axs[0].grid(True, axis='y')

# Original G PDF
axs[1].plot(x, pdf_G, color='g', linewidth=1)
axs[1].set_title("Original G")
axs[1].set_xticks(range(0, 256, 15))
axs[1].grid(True, axis='y')

# Original B PDF
axs[2].plot(x, pdf_B, color='b', linewidth=1)
axs[2].set_title("Original B")
axs[2].set_xticks(range(0, 256, 15))
axs[2].grid(True, axis='y')

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Output R PDF
axs[0].plot(x, pdf_RO, color='r', linewidth=1)
axs[0].set_title("Output R")
axs[0].set_xticks(range(0, 256, 15))
axs[0].grid(True, axis='y')

# Output G PDF
axs[1].plot(x, pdf_GO, color='g', linewidth=1)
axs[1].set_title("Output G")
axs[1].set_xticks(range(0, 256, 15))
axs[1].grid(True, axis='y')

# Output B PDF
axs[2].plot(x, pdf_BO, color='b', linewidth=1)
axs[2].set_title("Output B")
axs[2].set_xticks(range(0, 256, 15))
axs[2].grid(True, axis='y')

plt.tight_layout()
plt.show()