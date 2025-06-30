#include "hist_func.h"

void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF);

int main() {

	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat equalized_RGB = input.clone();

	// PDF or transfer function txt files
	FILE *f_equalized_PDF_RGB, *f_PDF_RGB;
	FILE *f_trans_func_eq_RGB;
	FILE* f_trans_func_R, * f_trans_func_G, * f_trans_func_B;
	FILE* o_r, * o_g, * o_b;
	FILE* out_r, * out_g, * out_b;
	
	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_equalized_PDF_RGB, "equalized_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_eq_RGB, "trans_func_eq_RGB.txt", "w+");

	fopen_s(&f_trans_func_R, "trans_func_R.txt", "w+");
	fopen_s(&f_trans_func_G, "trans_func_G.txt", "w+");
	fopen_s(&f_trans_func_B, "trans_func_B.txt", "w+");

	fopen_s(&o_r, "original_R.txt", "w+");
	fopen_s(&o_g, "original_G.txt", "w+");
	fopen_s(&o_b, "original_B.txt", "w+");

	fopen_s(&out_r, "output_R.txt", "w+");
	fopen_s(&out_g, "output_G.txt", "w+");
	fopen_s(&out_b, "output_B.txt", "w+");

	float **PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float **CDF_RGB = cal_CDF_RGB(input);	// CDF of Input image(RGB) : [L][3]

	G trans_func_eq_RGB[L][3] = { 0 };		// transfer function

	// histogram equalization on RGB image
	hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);
	// equalized PDF (RGB)
	float** equalized_PDF_RGB = cal_PDF_RGB(equalized_RGB);
	
	for (int i = 0; i < L; i++) {
		for (int c = 0; c < 3; c++) {
			fprintf(f_equalized_PDF_RGB, "%d\t%f\n", i + c * L, equalized_PDF_RGB[i][c]);
			fprintf(f_PDF_RGB, "%d\t%f\n", i + c * L, PDF_RGB[i][c]);
			fprintf(f_trans_func_eq_RGB, "%d\t%d\n", i + c * L, trans_func_eq_RGB[i][c]);
		}
	}

	// txt 파일로 만들기
	for (int i = 0; i < L; i++) {
		fprintf(f_trans_func_R, "%d\t%d\n", i, trans_func_eq_RGB[i][2]); // R
		fprintf(f_trans_func_G, "%d\t%d\n", i, trans_func_eq_RGB[i][1]); // G
		fprintf(f_trans_func_B, "%d\t%d\n", i, trans_func_eq_RGB[i][0]); // B

		fprintf(o_r, "%d\t%d\n", i, PDF_RGB[i][2]); //R
		fprintf(o_g, "%d\t%d\n", i, PDF_RGB[i][1]); //G
		fprintf(o_b, "%d\t%d\n", i, PDF_RGB[i][0]); //B

		fprintf(out_r, "%d\t%d\n", i, equalized_PDF_RGB[i][2]); //R
		fprintf(out_g, "%d\t%d\n", i, equalized_PDF_RGB[i][1]); //G
		fprintf(out_b, "%d\t%d\n", i, equalized_PDF_RGB[i][0]); //B
	}


	// memory release
	free(PDF_RGB);
	free(CDF_RGB);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_eq_RGB);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
	imshow("Equalized_RGB", equalized_RGB);
	imwrite("Equalized_RGB_image.png", equalized_RGB);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization on 3 channel image
void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF) {
	// compute transfer function
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < L; i++) {
			trans_func[i][c] = (G)((L - 1) * CDF[i][c]);
		}
	}

	// perform the transfer function
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			for (int c = 0; c < 3; c++) {
				equalized.at<Vec3b>(i, j)[c] = trans_func[input.at<Vec3b>(i, j)[c]][c];
			}
		}
	}
}