#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(const Mat& vec1, const Mat& vec2) {
    double sum = 0.0;
    for (int i = 0; i < vec1.cols; i++) {
        sum += pow(vec1.at<float>(0, i) - vec2.at<float>(0, i), 2);
    }
    return sqrt(sum);
}

int nearestNeighbor(const Mat& vec, const vector<KeyPoint>& keypoints, const Mat& descriptors) {
    int neighbor = -1;
    double minDist = 1e6;
    for (int i = 0; i < descriptors.rows; i++) {
        double temp = euclidDistance(vec, descriptors.row(i));
        if (temp < minDist) {
            minDist = temp;
            neighbor = i;
        }
    }
    return neighbor;
}

void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
    vector<KeyPoint>& keypoints2, Mat& descriptors2,
    vector<Point2f>& srcPoints, vector<Point2f>& dstPoints,
    bool crossCheck, bool ratio_threshold) {
    for (int i = 0; i < descriptors1.rows; i++) {
        Mat desc1 = descriptors1.row(i);
        int nn = nearestNeighbor(desc1, keypoints2, descriptors2);
        if (nn < 0 || nn >= keypoints2.size()) continue;

        double minDist = 1e6, secondMinDist = 1e6;
        for (int j = 0; j < descriptors2.rows; j++) {
            double temp = euclidDistance(desc1, descriptors2.row(j));
            if (temp < minDist) {
                secondMinDist = minDist;
                minDist = temp;
            }
            else if (temp < secondMinDist) {
                secondMinDist = temp;
            }
        }

        if (ratio_threshold && minDist / secondMinDist >= RATIO_THR) continue;

        if (crossCheck) {
            Mat desc2 = descriptors2.row(nn);
            int reverseNN = nearestNeighbor(desc2, keypoints1, descriptors1);
            if (reverseNN != i) continue;
        }

        srcPoints.push_back(keypoints1[i].pt);
        dstPoints.push_back(keypoints2[nn].pt);
    }
}

template <typename T>
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points) {
    Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
    Mat b(2 * number_of_points, 1, CV_32F);
    for (int i = 0; i < number_of_points; i++) {
        M.at<T>(2 * i, 0) = ptl_x[i];
        M.at<T>(2 * i, 1) = ptl_y[i];
        M.at<T>(2 * i, 2) = 1;
        M.at<T>(2 * i + 1, 3) = ptl_x[i];
        M.at<T>(2 * i + 1, 4) = ptl_y[i];
        M.at<T>(2 * i + 1, 5) = 1;
        b.at<T>(2 * i) = ptr_x[i];
        b.at<T>(2 * i + 1) = ptr_y[i];
    }
    Mat M_trans, temp, affineM;
    transpose(M, M_trans);
    invert(M_trans * M, temp);
    affineM = temp * M_trans * b;
    return affineM;
}

Mat ransac_affine(const int ptl_x[], const int ptl_y[], const int ptr_x[], const int ptr_y[], int total_points, int sample_size, int iterations, float threshold) {
    Mat best_affine;
    int max_inliers = 0;
    std::vector<int> all_indices(total_points);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    for (int iter = 0; iter < iterations; iter++) {
        std::shuffle(all_indices.begin(), all_indices.end(), g);
        int lx[10], ly[10], rx[10], ry[10];
        for (int i = 0; i < sample_size; i++) {
            lx[i] = ptl_x[all_indices[i]];
            ly[i] = ptl_y[all_indices[i]];
            rx[i] = ptr_x[all_indices[i]];
            ry[i] = ptr_y[all_indices[i]];
        }
        Mat affine = cal_affine<float>(lx, ly, rx, ry, sample_size);
        int inliers = 0;
        for (int i = 0; i < total_points; i++) {
            float x = ptl_x[i], y = ptl_y[i];
            float x_p = affine.at<float>(0) * x + affine.at<float>(1) * y + affine.at<float>(2);
            float y_p = affine.at<float>(3) * x + affine.at<float>(4) * y + affine.at<float>(5);
            float dx_ = ptr_x[i] - x_p;
            float dy_ = ptr_y[i] - y_p;
            float error = dx_ * dx_ + dy_ * dy_;
            if (error < threshold * threshold)
                inliers++;
        }
        if (inliers > max_inliers) {
            max_inliers = inliers;
            best_affine = affine.clone();
        }
    }
    return best_affine;
}

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha) {
    for (int i = 0; i < I1.rows; i++) {
        for (int j = 0; j < I1.cols; j++) {
            bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0);
            if (cond_I2)
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);
            else
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);
        }
    }
}

int main() {
    Mat input1 = imread("Img01.jpg", IMREAD_COLOR);
    Mat input2 = imread("Img02.jpg", IMREAD_COLOR);
    if (!input1.data || !input2.data) {
        cout << "Could not open one of the input images." << endl;
        return -1;
    }

    Mat input1_gray, input2_gray;
    cvtColor(input1, input1_gray, COLOR_BGR2GRAY);
    cvtColor(input2, input2_gray, COLOR_BGR2GRAY);

    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    sift->detectAndCompute(input1_gray, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(input2_gray, noArray(), keypoints2, descriptors2);

    vector<Point2f> srcPoints, dstPoints;
    findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints, dstPoints, true, true);

    if (srcPoints.size() < 4) {
        cout << "Not enough matches." << endl;
        return -1;
    }

    int total = srcPoints.size();
    int sx[100], sy[100], dx[100], dy[100];
    for (int i = 0; i < total; i++) {
        sx[i] = srcPoints[i].x;
        sy[i] = srcPoints[i].y;
        dx[i] = dstPoints[i].x;
        dy[i] = dstPoints[i].y;
    }

    Mat A = ransac_affine(sx, sy, dx, dy, total, 4, 300, 3.0);

    // bounding box 계산
    int bound_l = 0, bound_u = 0, bound_r = input1.cols + input2.cols, bound_b = max(input1.rows, input2.rows);
    Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));

    // inverse warping with bilinear interpolation
    for (int i = bound_u; i <= bound_b; i++) {
        for (int j = bound_l; j <= bound_r; j++) {
            float x = A.at<float>(0) * j + A.at<float>(1) * i + A.at<float>(2);
            float y = A.at<float>(3) * j + A.at<float>(4) * i + A.at<float>(5);
            float y1 = floor(y), y2 = ceil(y), x1 = floor(x), x2 = ceil(x);
            float mu = y - y1, lambda = x - x1;
            if (x1 >= 0 && x2 < input2.cols && y1 >= 0 && y2 < input2.rows)
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3f>(y2, x2) + (1 - mu) * input2.at<Vec3f>(y1, x2)) +
                (1 - lambda) * (mu * input2.at<Vec3f>(y2, x1) + (1 - mu) * input2.at<Vec3f>(y1, x1));
        }
    }

    input1.convertTo(input1, CV_32FC3, 1.0 / 255);
    blend_stitching(input1, input2, I_f, bound_l, bound_u, 0.5);

    imshow("Stitched Result", I_f);
    I_f.convertTo(I_f, CV_8UC3, 255);
    imwrite("stitched_output.jpg", I_f);
    waitKey(0);
    return 0;
}
