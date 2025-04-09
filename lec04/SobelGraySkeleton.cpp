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

Mat sobelfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, COLOR_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = sobelfilter(input_gray); //Boundary process: mirroring

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	Mat kernel;

	int tempa;
	int tempb;
	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Sobel kernel
	int Sx[3][3] = { {-1, 0, 1},
					 {-2, 0, 2},
					 {-1, 0, 1} };

	int Sy[3][3] = { {-1, -2, -1},
					 { 0,  0,  0},
					 { 1,  2,  1} };

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Ix = 0;
			int ly = 0;

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

					Ix += input.at<G>(tempa, tempb) * Sx[a + n][b + n];
					ly += input.at<G>(tempa, tempb) * Sy[a + n][b + n];

					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
				}
			}

			int Mxy = (int)sqrt(Ix * Ix + ly * ly);
			if (Mxy > 255) Mxy = 255;
			if (Mxy < 0) Mxy = 0;

			output.at<G>(i, j) = (G)Mxy;
		}
	}
	return output;
}