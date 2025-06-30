#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256		// # of intensity levels
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

// generate PDF for single channel image
float *cal_PDF(Mat &input) {

	int count[L] = { 0 };
	float *PDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute PDF
	for (int i = 0; i < L; i++)
		PDF[i] = (float)count[i] / (float)(input.rows * input.cols);

	return PDF;
}

// generate PDF for color image
float **cal_PDF_RGB(Mat &input) {

	int count[L][3] = { 0 };
	float **PDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		PDF[i] = (float*)calloc(3, sizeof(float));

	//Count
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			for (int c = 0; c < 3; c++) {
				count[input.at<Vec3b>(i, j)[c]][c]++;
			}
		}
	}

	//Compute PDF
	for (int i = 0; i < L; i++) {
		for (int c = 0; c < 3; c++) {
			PDF[i][c] = (float)count[i][c] / (float)(input.rows * input.cols);
		}
	}

	return PDF;
}

// generate CDF for single channel image
float *cal_CDF(Mat &input) {

	int count[L] = { 0 };
	float *CDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute CDF
	for (int i = 0; i < L; i++) {
		CDF[i] = (float)count[i] / (float)(input.rows * input.cols);

		if (i != 0)
			CDF[i] += CDF[i - 1];
	}

	return CDF;
}

// generate CDF for color image
float **cal_CDF_RGB(Mat &input) {

	int count[L][3] = { 0 };
	float **CDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		CDF[i] = (float*)calloc(3, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			for (int c = 0; c < 3; c++) {
				count[input.at<Vec3b>(i, j)[c]][c]++;
			}
		}
	}

	// Compute CDF
	for (int i = 0; i < L; i++) {
		for (int c = 0; c < 3; c++) {
			CDF[i][c] = (float)count[i][c] / (float)(input.rows * input.cols);

			if (i != 0)
				CDF[i][c] += CDF[i - 1][c];
		}
	}

	return CDF;
}