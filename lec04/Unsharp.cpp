#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
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
	Mat input_gray, output;
	float k = 0.7f;

	if (!input.data) {
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	cvtColor(input, input_gray, COLOR_RGB2GRAY);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	output = UnsharpMask(input_gray, 1, 1.0, 1.0, "adjustkernel", k); // "zero-paddle", "mirroring", "adjustkernel"
	
	tm.stop();

	namedWindow("Unsharp Masking Filter", WINDOW_AUTOSIZE);
	imshow("Unsharp Masking Filter", output);
	std::cout << "소요시간: " << tm.getTimeMilli() << " ms" << std::endl;

	waitKey(0);
	return 0;
}

// 1D Gaussian 커널 생성
std::vector<float> getGaussianKernel(int n, float sigma) {
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


Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) {
	int row = input.rows;
	int col = input.cols;


	std::vector<float> kernelY = getGaussianKernel(n, sigmaT);
	std::vector<float> kernelX = getGaussianKernel(n, sigmaS);

	int tempa, tempb;

	Mat temp = Mat::zeros(row, col, input.type());
	Mat L = Mat::zeros(row, col, input.type());
	Mat output = Mat::zeros(row, col, input.type());

	// X 방향 필터링
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum = 0.0;

			if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0, sum2 = 0.0;
				for (int b = -n; b <= n; b++) {
					if (j + b >= 0 && j + b < col) {
						sum1 += kernelX[b + n] * (float)(input.at<G>(i, j + b));
						sum2 += kernelX[b + n];
					}
				}
				sum = (sum2 != 0.0f) ? sum1 / sum2 : 0.0f;
			}
			else {
				for (int b = -n; b <= n; b++) {
					if (!strcmp(opt, "zero-paddle")) {
						if (j + b >= 0 && j + b < col)
							sum += kernelX[b + n] * input.at<G>(i, j + b);
					}
					else if (!strcmp(opt, "mirroring")) {
						if (j + b < 0)
							tempb = -(j + b);
						else if (j + b >= col)
							tempb = j - b;
						else
							tempb = j + b;

						sum += kernelX[b + n] * input.at<G>(i, tempb);
					}
				}
			}
			temp.at<G>(i, j) = (G)sum;
		}
	}

	// Y 방향 필터링
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum = 0.0;

			if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0, sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					if (i + a >= 0 && i + a < row) {
						sum1 += kernelY[a + n] * (float)(temp.at<G>(i + a, j));
						sum2 += kernelY[a + n];
					}
				}
				sum = (sum2 != 0.0f) ? sum1 / sum2 : 0.0f;
			}
			else {
				for (int a = -n; a <= n; a++) {
					if (!strcmp(opt, "zero-paddle")) {
						if (i + a >= 0 && i + a < row)
							sum += kernelY[a + n] * temp.at<G>(i + a, j);
					}
					else if (!strcmp(opt, "mirroring")) {
						if (i + a < 0)
							tempa = -(i + a);
						else if (i + a >= row)
							tempa = i - a;
						else
							tempa = i + a;

						sum += kernelY[a + n] * temp.at<G>(tempa, j);
					}
				}
			}
			L.at<G>(i, j) = (G)sum;
		}
	}
	
	Mat input_f, L_f, sharp_f;

	
	input.convertTo(input_f, CV_32F);

	L.convertTo(L_f, CV_32F);

	Mat kL = L_f * k;

	Mat df = input_f - kL;

	sharp_f = df * (1.0f / (1.0f - k));

	sharp_f.convertTo(output, CV_8U);


	return output;
}