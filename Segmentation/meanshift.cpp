#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray, input_gray_3;
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	cvtColor(input_gray, input_gray_3, COLOR_GRAY2BGR);

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input_gray_3);

    pyrMeanShiftFiltering(input_gray_3, output, 31, 20, 3);
	imshow("Meanshift Gray", output);

	waitKey();

	return 0;
}