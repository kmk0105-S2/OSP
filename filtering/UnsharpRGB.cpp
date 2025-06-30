#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

TickMeter tm;

// Image Type
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

// 함수 선언
Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);
std::vector<float> getGaussianKernel(int n, float sigma);

int main() {
	tm.start();

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat output;
	float k = 0.5f;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = UnsharpMask(input, 1, 1.0, 1.0, "adjustkernel", k); //Boundary process: zero-paddle, mirroring, adjustkernel

	tm.stop();

	namedWindow("Unsharp Masking Filter RGB", WINDOW_AUTOSIZE);
	imshow("Unsharp Masking Filter RGB", output);

	std::cout << "소요시간: " << tm.getTimeMilli() << " ms" << std::endl;

	waitKey(0);

	return 0;
}

// 1D Gaussian 커널 생성
std::vector<float> getGaussianKernelRGB(int n, float sigma) {
	std::vector<float> kernel(2 * n + 1);
	float sum = 0.0;

	for (int i = -n; i <= n; i++) {
		kernel[i + n] = exp(-(i * i) / (2 * sigma * sigma));
		sum += kernel[i + n];
	}
	for (int i = 0; i < 2 * n + 1; i++) {
		kernel[i] /= sum;
	}
	return kernel;
}

// Separable Gaussian Filter
Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) {
	int row = input.rows;
	int col = input.cols;

	std::vector<float> kernelY = getGaussianKernelRGB(n, sigmaT);
	std::vector<float> kernelX = getGaussianKernelRGB(n, sigmaS);

	int tempa, tempb;

	Mat temp = Mat::zeros(row, col, input.type());
	Mat L = Mat::zeros(row, col, input.type());
	Mat output = Mat::zeros(row, col, input.type());

	// X 방향 필터링
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sumR = 0.0, sumG = 0.0, sumB = 0.0;
			float weightSum = 0.0;

			for (int b = -n; b <= n; b++) {
				int x = j + b;
				int tempb;

				if (!strcmp(opt, "zero-paddle")) {
					if (x < 0 || x >= col) continue;
					tempb = x;
				}
				else if (!strcmp(opt, "mirroring")) {
					if (x < 0) tempb = -x;
					else if (x >= col) tempb = col - (x - col) - 1;
					else tempb = x;
				}
				else if (!strcmp(opt, "adjustkernel")) {
					if (x < 0 || x >= col) continue;
					tempb = x;
				}
				else continue;

				Vec3b pixel = input.at<Vec3b>(i, tempb);
				float weight = kernelX[b + n];

				sumR += weight * pixel[0];
				sumG += weight * pixel[1];
				sumB += weight * pixel[2];

				if (!strcmp(opt, "adjustkernel"))
					weightSum += weight;
			}

			if (!strcmp(opt, "adjustkernel") && weightSum != 0.0f) {
				sumR /= weightSum;
				sumG /= weightSum;
				sumB /= weightSum;
			}

			temp.at<Vec3b>(i, j) = Vec3b((uchar)sumR, (uchar)sumG, (uchar)sumB);
		}
	}

	// Y 방향 필터링
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sumR = 0.0, sumG = 0.0, sumB = 0.0;
			float weightSum = 0.0;

			for (int a = -n; a <= n; a++) {
				int y = i + a;
				int tempa;

				if (!strcmp(opt, "zero-paddle")) {
					if (y < 0 || y >= row) continue;
					tempa = y;
				}
				else if (!strcmp(opt, "mirroring")) {
					if (i + a > row - 1) {
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}

				}
				else if (!strcmp(opt, "adjustkernel")) {
					if (y < 0 || y >= row) continue;
					tempa = y;
				}
				else continue;

				Vec3b pixel = temp.at<Vec3b>(tempa, j);
				float weight = kernelY[a + n];

				sumR += weight * pixel[0];
				sumG += weight * pixel[1];
				sumB += weight * pixel[2];

				if (!strcmp(opt, "adjustkernel"))
					weightSum += weight;
			}

			if (!strcmp(opt, "adjustkernel") && weightSum != 0.0f) {
				sumR /= weightSum;
				sumG /= weightSum;
				sumB /= weightSum;
			}

			L.at<Vec3b>(i, j) = Vec3b((uchar)sumR, (uchar)sumG, (uchar)sumB);
		}
	}

	Mat input_f, L_f, sharp_f;


	input.convertTo(input_f, CV_32FC3);

	L.convertTo(L_f, CV_32FC3);

	Mat kL = L_f * k;

	Mat df = input_f - kL;

	sharp_f = df * (1.0f / (1.0f - k));

	sharp_f.convertTo(output, CV_8UC3);


	return output;
}