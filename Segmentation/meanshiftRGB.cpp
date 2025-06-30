#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


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

    pyrMeanShiftFiltering(input, output, 31, 20, 3);
	imshow("Meanshift", output);

	waitKey();

	return 0;
}