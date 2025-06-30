#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;


// Note that this code is for the case when an input data is a color value.
int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray;
	Mat output;
	cvtColor(input, input_gray, COLOR_RGB2GRAY); // Converting image to gray


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input_gray);

	
	int row = input_gray.rows;
	int col = input_gray.cols;

	Mat samples(input_gray.rows * input_gray.cols, 3, CV_32F);
	float sigma = 10.0;
	for (int y = 0; y < row; y++) {
		for (int x = 0; x < col; x++) {
			samples.at<float>(y * col + x, 0) = (float)input_gray.at<uchar>(y, x);
			samples.at<float>(y * col + x, 1) = (float)x / sigma;
			samples.at<float>(y * col + x, 2) = (float)y / sigma;
		}
	}

	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(input_gray.size(), input_gray.type());
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
		{
			int cluster_idx = labels.at<int>(y * col + x, 0);
			new_image.at<uchar>(y, x) = (uchar)(centers.at<float>(cluster_idx, 0));
			
		}
	imshow("clustered image", new_image);

	Mat color_label_map;
	applyColorMap(new_image, color_label_map, COLORMAP_JET);
	imshow("Color label map", color_label_map);

	waitKey(0);

	return 0;
}

