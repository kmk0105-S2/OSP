#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

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

Mat laplacianfilterRGB(const Mat input);

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = laplacianfilterRGB(input); //Boundary process: mirroring

	namedWindow("Laplacian Filter RGB", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter RGB", output);


	waitKey(0);

	return 0;
}


Mat laplacianfilterRGB(const Mat input) {

	Mat kernel;

	int tempa;
	int tempb;
	int row = input.rows;
	int col = input.cols;
	int n = 1; // laplacian Filter Kernel N
	
	// laplacian kernel
	int L[3][3] = { {0, 1, 0},
					 {1, -4, 1},
					 {0, 1, 0} };

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int value[3] = { 0, 0, 0 };

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

					for (int c = 0; c < 3; c++) {
						value[c] += pixel[c] * L[a + n][b + n];
					}
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
				}
			}

			Vec3b result;

			for (int c = 0; c < 3; c++) {
				value[c] = abs(value[c]) * 3;
				if (value[c] > 255) value[c] = 255;
				result[c] = (uchar)value[c];
			}
			output.at<Vec3b>(i, j) = result;
		}
	}
	return output;
}