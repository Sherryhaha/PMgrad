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

    void pmgrad(Mat&A,Mat&A1,Mat&B,double k);
    void converttochar(Mat&A,Mat&B);
    void gradz(Mat&A,Mat&B);
    void gradu(Mat&A,Mat&B);
};

#endif //PMGRADENHENCE_PMGRADENHENCE_H
