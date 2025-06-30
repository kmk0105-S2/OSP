#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

TickMeter tm;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
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

Mat gaussianfilter_RGB(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {
	tm.start();

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat output;



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = gaussianfilter_RGB(input, 1, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel

	tm.stop();

	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter RGB", output);
	std::cout << "소요시간: " << tm.getTimeMilli() << " ms" << std::endl;

	waitKey(0);

	return 0;
}


Mat gaussianfilter_RGB(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);


	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(opt, "zero-paddle")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {

							Vec3b pixel = input.at<Vec3b>(i + a, j + b);
							float weight = kernel.at<float>(a + n, b + n);

							sum1_r += weight * pixel[0];
							sum1_g += weight * pixel[1];
							sum1_b += weight * pixel[2];
						}

					}
				}

				output.at<Vec3b>(i, j) = Vec3b((uchar)sum1_r, (uchar)sum1_g, (uchar)sum1_b);
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if (i + a > row - 1) { // 커널이 아래 경계를 벗어남
							tempa = i - a;
						}
						else if (i + a < 0) { //커널이 위쪽 경계를 벗어남
							tempa = -(i + a);
						}
						else { //이미지 안쪽일 때 그대로
							tempa = i + a;
						}
						if (j + b > col - 1) { // 커널이 오른쪽 경계를 벗어남
							tempb = j - b;
						}
						else if (j + b < 0) { //커널이 왼쪽 경게를 벗어남
							tempb = -(j + b);
						}
						else { //이미지 안쪽일 때 그대로
							tempb = j + b;
						}
						Vec3b pixel = input.at<Vec3b>(tempa, tempb);
						float weight = kernel.at<float>(a + n, b + n);

						sum1_r += weight * pixel[0];
						sum1_g += weight * pixel[1];
						sum1_b += weight * pixel[2];
					}
				}
				output.at<Vec3b>(i, j) = Vec3b((uchar)sum1_r, (uchar)sum1_g, (uchar)sum1_b);
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;

				float weight_sum = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							Vec3b pixel = input.at<Vec3b>(i + a, j + b);
							float weight = kernel.at<float>(a + n, b + n);

							sum1_r += weight * pixel[0];
							sum1_g += weight * pixel[1];
							sum1_b += weight * pixel[2];

							weight_sum += weight;
						}

					}
				}
				output.at<Vec3b>(i, j) = Vec3b(
					(uchar)(sum1_r / weight_sum),
					(uchar)(sum1_g / weight_sum),
					(uchar)(sum1_b / weight_sum)
				);
			}
		}
	}
	return output;
}