#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;


// Note that this code is for the case when an input data is a color value.
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

	int row = input.rows;
	int col = input.cols;
	float sigma = 10.0f;
	Mat samples(input.rows * input.cols, 5, CV_32F);
	for (int y = 0; y < row; y++) {
		for (int x = 0; x < col; x++) {
			Vec3b pixel = input.at<Vec3b>(y, x);
			samples.at<float>(y * col + x, 0) = pixel[0];
			samples.at<float>(y * col + x, 1) = pixel[1];
			samples.at<float>(y * col + x, 2) = pixel[2];
			samples.at<float>(y * col + x, 3) = (float)x/sigma;
			samples.at<float>(y * col + x, 4) = (float)y/sigma;
		}
	}

	// Clustering is performed for each channel (RGB)
	// Note that the intensity value is not normalized here (0~1). You should normalize both intensity and position when using them simultaneously.
	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(input.size(), input.type());
	for (int y = 0; y < row; y++)
		for (int x = 0; x < col; x++)
		{
			int cluster_idx = labels.at<int>(y*col + x, 0);

			new_image.at<Vec3b>(y, x)[0] = (uchar)centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = (uchar)centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = (uchar)centers.at<float>(cluster_idx, 2);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
		}
	imshow("clustered image", new_image);

	Mat color_label_map;
	applyColorMap(new_image, color_label_map, COLORMAP_JET);
	imshow("Color label map", color_label_map);

	waitKey(0);

	return 0;
}

