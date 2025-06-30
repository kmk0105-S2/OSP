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

	Mat refer = imread("Equalized_YUV_image.png", IMREAD_COLOR); //refer 이미지 불러오기
	Mat refer_YUV;

	cvtColor(refer, refer_YUV, COLOR_BGR2YUV); //YUV 이미지로 변환

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
	float** PDF_RGB_rf = cal_PDF_RGB(refer); // refer 이미지의 PDF RGB 구하기
	float* CDF_YUV_rf = cal_CDF(Y_rf); // refer 이미지의 Y 채널의 CDF 구하기

	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_matching_PDF_YUV, "matching_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_mat, "trans_func_mat.txt", "w+");

	G match_func[L] = { 0 };  // refer 함수

	hist_matching(match_func, CDF_YUV, CDF_YUV_rf);

	// 실제 Y 채널 픽셀 매칭
	Mat Y_matched = Y.clone();
	for (int i = 0; i < Y.rows; i++) {
		for (int j = 0; j < Y.cols; j++) {
			Y_matched.at<G>(i, j) = match_func[Y.at<G>(i, j)];
		}
	}

	// Y 채널 교체
	in_channels[0] = Y_matched;

	// YUV 재합성 → RGB
	merge(in_channels, 3, matched_YUV);

	// YUV -> RGB
	Mat output;
	cvtColor(matched_YUV, output, COLOR_YUV2BGR);

	// PDF 계산
	float* PDF = cal_PDF(Y);           // 원본 Y
	float* matched_PDF = cal_PDF(Y_matched); // 매칭된 Y

	// 파일 출력
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