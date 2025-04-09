#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear"); //������ ����

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	// inverse warping
	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2; //ȸ�� ������ ��ǥ* ȸ�� ��� => ���� ��ǥ(float)�� ��ġ ã��
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) { //float���� �ٲ� ��ǥ�� �̿��ؼ� interpolation�ϱ�!! (�� �� �޲ٱ�)
				if (!strcmp(opt, "nearest")) { //�ֺ� �� ����
					int nearX = round(x);
					int nearY = round(y);

					output.at<T>(i, j) = input.at<T>(nearY, nearX); //opencv ��ǥ��!!

				}
				else if (!strcmp(opt, "bilinear")) { //������
					int x1 = floor(x);
					int x2 = ceil(x);
					int y1 = floor(y);
					int y2 = ceil(y);

					float mu = x - x1;
					float lam = y - y1;

					T f_x_y1 = mu * input.at<T>(y2, x2) + (1 - mu) * input.at<T>(y2, x1);
					T f_x1_y = mu * input.at<T>(y1, x2) + (1 - mu) * input.at<T>(y1, x1);
					T f_x1_y1 = lam * f_x_y1 + (1 - lam) * f_x1_y;

					output.at<T>(i, j) = f_x1_y1;

				}
			}
		}
	}

	return output;
}