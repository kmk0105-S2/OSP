#include "hist_func.h"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_matching(G* match_func, float* CDF, float* CDF_refer);

int main() {

	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat matched_YUV;

	cvtColor(input, matched_YUV, COLOR_BGR2YUV);	// RGB -> YUV

	//split channel
	Mat in_channels[3];
	split(matched_YUV, in_channels);
	Mat Y = in_channels[0];

	Mat refer = imread("Equalized_YUV_image.png", IMREAD_COLOR); //refer �̹��� �ҷ�����
	Mat refer_YUV;

	cvtColor(refer, refer_YUV, COLOR_BGR2YUV); //YUV �̹����� ��ȯ

	//YUV split channel
	Mat out_channels[3];
	split(refer_YUV, out_channels);
	Mat Y_rf = out_channels[0];

	// PDF or transfer function txt files
	FILE* f_PDF;
	FILE* f_matching_PDF_YUV;
	FILE* f_trans_func_mat;

	float** PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float* CDF_YUV = cal_CDF(Y);	// CDF of Y channel
	float** PDF_RGB_rf = cal_PDF_RGB(refer); // refer �̹����� PDF RGB ���ϱ�
	float* CDF_YUV_rf = cal_CDF(Y_rf); // refer �̹����� Y ä���� CDF ���ϱ�

	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_matching_PDF_YUV, "matching_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_mat, "trans_func_mat.txt", "w+");

	G match_func[L] = { 0 };  // refer �Լ�

	hist_matching(match_func, CDF_YUV, CDF_YUV_rf);

	// ���� Y ä�� �ȼ� ��Ī
	Mat Y_matched = Y.clone();
	for (int i = 0; i < Y.rows; i++) {
		for (int j = 0; j < Y.cols; j++) {
			Y_matched.at<G>(i, j) = match_func[Y.at<G>(i, j)];
		}
	}

	// Y ä�� ��ü
	in_channels[0] = Y_matched;

	// YUV ���ռ� �� RGB
	merge(in_channels, 3, matched_YUV);

	// YUV -> RGB
	Mat output;
	cvtColor(matched_YUV, output, COLOR_YUV2BGR);

	// PDF ���
	float* PDF = cal_PDF(Y);           // ���� Y
	float* matched_PDF = cal_PDF(Y_matched); // ��Ī�� Y

	// ���� ���
	for (int i = 0; i < L; i++) {
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_matching_PDF_YUV, "%d\t%f\n", i, matched_PDF[i]);
		fprintf(f_trans_func_mat, "%d\t%d\n", i, match_func[i]);
	}

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Matching_YUV", WINDOW_AUTOSIZE);
	imshow("Matching_YUV", output);
	imwrite("matching_image_YUV.png", output);

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