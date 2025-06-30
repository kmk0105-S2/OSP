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

Mat laplacianfilter(const Mat input);

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
	output = laplacianfilter(input_gray); //Boundary process: mirroring

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);


	waitKey(0);

	return 0;
}


Mat laplacianfilter(const Mat input) {

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
			int value = 0;

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

					value += input.at<G>(tempa, tempb) * L[a + n][b + n];

					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
				}
			}
			if (value > 255) value = 255;
			if (value < 0) value = 0;

			output.at<G>(i, j) = (G)value;
		}
	}
	return output;
}