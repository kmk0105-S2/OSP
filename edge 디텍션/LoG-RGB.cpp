#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat get_Laplacian_Kernel();
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Laplacianfilter(const Mat input);
Mat Mirroring(const Mat input, int n);

int main() {
    Mat input = imread("lena.jpg", IMREAD_COLOR);

    if (!input.data) {
        printf("Could not open\n");
        return -1;
    }

    // Parameters
    int window_radius = 2;
    double sigma_t = 2.0;
    double sigma_s = 2.0;

    input.convertTo(input, CV_64FC3, 1.0 / 255);  // Normalize RGB

    Mat h_f = Gaussianfilter(input, window_radius, sigma_t, sigma_s);
    Mat Laplacian = Laplacianfilter(h_f);

    
    Mat input_disp, h_f_disp, Laplacian_disp;
    input.convertTo(input_disp, CV_8UC3, 255);
    h_f.convertTo(h_f_disp, CV_8UC3, 255);

    
    vector<Mat> channels(3), abs_channels(3);
    split(Laplacian, channels);
    for (int i = 0; i < 3; ++i) {
        abs_channels[i] = abs(channels[i]);
        normalize(abs_channels[i], abs_channels[i], 0, 255, NORM_MINMAX);
    }
    merge(abs_channels, Laplacian_disp);
    Laplacian_disp.convertTo(Laplacian_disp, CV_8UC3);

    imshow("RGB", input_disp);
    imshow("Gaussian blur", h_f_disp);
    imshow("Laplacian filter", Laplacian_disp);

    waitKey(0);
    return 0;
}

Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s) {
    int row = input.rows;
    int col = input.cols;
    Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
    Mat output = Mat::zeros(row, col, input.type());
    Mat input_mirror = Mirroring(input, n);

    for (int i = n; i < row + n; i++) {
        for (int j = n; j < col + n; j++) {
            Vec3d sum = Vec3d(0, 0, 0);
            for (int a = -n; a <= n; a++) {
                for (int b = -n; b <= n; b++) {
                    Vec3d pixel = input_mirror.at<Vec3d>(i + a, j + b);
                    double weight = kernel.at<double>(a + n, b + n);
                    sum += weight * pixel;
                }
            }
            output.at<Vec3d>(i - n, j - n) = sum;
        }
    }

    return output;
}

Mat Laplacianfilter(const Mat input) {
    int row = input.rows;
    int col = input.cols;
    Mat kernel = get_Laplacian_Kernel();
    Mat output = Mat::zeros(row, col, input.type());
    int n = 1;
    Mat input_mirror = Mirroring(input, n);

    for (int i = n; i < row + n; i++) {
        for (int j = n; j < col + n; j++) {
            Vec3d sum = Vec3d(0, 0, 0);
            for (int a = -n; a <= n; a++) {
                for (int b = -n; b <= n; b++) {
                    Vec3d pixel = input_mirror.at<Vec3d>(i + a, j + b);
                    double weight = kernel.at<double>(a + n, b + n);
                    sum += weight * pixel;
                }
            }
            output.at<Vec3d>(i - n, j - n) = sum;
        }
    }

    return output;
}

Mat Mirroring(const Mat input, int n) {
    int row = input.rows;
    int col = input.cols;
    Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type());

    for (int i = n; i < row + n; i++) {
        for (int j = n; j < col + n; j++) {
            input2.at<Vec3d>(i, j) = input.at<Vec3d>(i - n, j - n);
        }
    }
    for (int i = n; i < row + n; i++) {
        for (int j = 0; j < n; j++)
            input2.at<Vec3d>(i, j) = input2.at<Vec3d>(i, 2 * n - j);
        for (int j = col + n; j < col + 2 * n; j++)
            input2.at<Vec3d>(i, j) = input2.at<Vec3d>(i, 2 * col - 2 + 2 * n - j);
    }
    for (int j = 0; j < col + 2 * n; j++) {
        for (int i = 0; i < n; i++)
            input2.at<Vec3d>(i, j) = input2.at<Vec3d>(2 * n - i, j);
        for (int i = row + n; i < row + 2 * n; i++)
            input2.at<Vec3d>(i, j) = input2.at<Vec3d>(2 * row - 2 + 2 * n - i, j);
    }

    return input2;
}

Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {
    int kernel_size = 2 * n + 1;
    Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
    double kernel_sum = 0.0;

    for (int i = -n; i <= n; i++) {
        for (int j = -n; j <= n; j++) {
            double value = exp(-((i * i) / (2.0 * sigma_t * sigma_t) + (j * j) / (2.0 * sigma_s * sigma_s)));
            kernel.at<double>(i + n, j + n) = value;
            kernel_sum += value;
        }
    }

    if (normalize) {
        kernel /= kernel_sum;
    }

    return kernel;
}

Mat get_Laplacian_Kernel() {
    Mat kernel = Mat::zeros(3, 3, CV_64F);
    kernel.at<double>(0, 1) = 1.0;
    kernel.at<double>(2, 1) = 1.0;
    kernel.at<double>(1, 0) = 1.0;
    kernel.at<double>(1, 2) = 1.0;
    kernel.at<double>(1, 1) = -4.0;
    return kernel;
}
