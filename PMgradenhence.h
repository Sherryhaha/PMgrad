//
// Created by sunguoyan on 16/3/24.
//

#ifndef PMGRADENHENCE_PMGRADENHENCE_H
#define PMGRADENHENCE_PMGRADENHENCE_H

#include <cv.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;

class PMgradenhence{
public:
    int X_image,Y_image;

    void max_min_grad(Mat&A,double*max,double*min);
    void gradenhence(Mat&A,Mat&B);
    void HistNormolize(Mat&pImg, Mat&pNormImg);
    void panduan(Mat&A);

    void pmgrad(Mat&A,Mat&A1,Mat&B,double k,double dt);
    void converttochar(Mat&A,Mat&B);
    void gradz(Mat&A,Mat&B);
    void gradu(Mat&A,Mat&B);

    void diffusion(Mat &A, Mat &B,double max,double min);

    void gradn(Mat &A,Mat &B,double*max,double*min);
    void grads(Mat &A,Mat &B,double*max,double*min);
    void gradw(Mat &A,Mat &B,double*max,double*min);
    void grade(Mat &A,Mat &B,double*max,double*min);
    void shang(Mat &A, double result);
};

#endif //PMGRADENHENCE_PMGRADENHENCE_H
