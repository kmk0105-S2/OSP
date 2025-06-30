#include "hist_func.h"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

void hist_matching(G* match_func, float* CDF, float* CDF_refer);

int main() {

	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat input_gray;

	cvtColor(input, input_gray, COLOR_RGB2GRAY);	// convert RGB to Grayscale

	Mat refer = imread("Equalized_YUV_image.png", IMREAD_COLOR); //refer 이미지 불러오기
	Mat refer_gray;

	cvtColor(refer, refer_gray, COLOR_RGB2GRAY); //gray 이미지로 변환

	Mat matched = input_gray.clone();

	// PDF or transfer function txt files
	FILE* f_PDF;
	FILE* f_matching_PDF_gray;
	FILE* f_trans_func_mat;

	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_matching_PDF_gray, "matching_PDF_gray.txt", "w+");
	fopen_s(&f_trans_func_mat, "trans_func_mat.txt", "w+");

	float* PDF = cal_PDF(input_gray);	// PDF of Input image(Grayscale) : [L]
	float* CDF = cal_CDF(input_gray);	// CDF of Input image(Grayscale) : [L]
	float* PDF_refer = cal_PDF(refer_gray); // refer 이미지의 PDF 구하기
	float* CDF_refer = cal_CDF(refer_gray); // refer 이미지의 CDF 구하기

	G match_func[L] = { 0 };  // refer 함수

	hist_matching(match_func, CDF, CDF_refer);

	for (int i = 0; i < input_gray.rows; i++) {
		for (int j = 0; j < input_gray.cols; j++) {
			matched.at<G>(i, j) = match_func[input_gray.at<G>(i, j)];
		}
	}

	float* matched_PDF_gray = cal_PDF(matched);
	
	for (int i = 0; i < L; i++) {
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_matching_PDF_gray, "%d\t%f\n", i, matched_PDF_gray[i]);
		fprintf(f_trans_func_mat, "%d\t%d\n", i, match_func[i]);
	}

	// memory release
	free(PDF_refer);
	free(CDF_refer);
	free(matched_PDF_gray);

	fclose(f_matching_PDF_gray);
	fclose(f_trans_func_mat);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Matching", WINDOW_AUTOSIZE);
	imshow("Matching", matched);
	imwrite("matching_image_gray.png", matched);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

void hist_matching(G* match_func, float* CDF, float* CDF_refer) {
	for (int i = 0; i < L; i++) {
		float t_r = CDF[i];
		int result = 0;
		float m_diff = 1.0f;

		for (int j = 0; j < L; j++) {
			float diff = fabs(t_r - CDF_refer[j]);
			if (diff < m_diff) {
				m_diff = diff;
				result = j;
			}
		}
		match_func[i] = result;
	}
}