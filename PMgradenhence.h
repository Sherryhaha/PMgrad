//
// Created by sunguoyan on 16/3/24.
//

#ifndef PMGRADENHENCE_PMGRADENHENCE_H
#define PMGRADENHENCE_PMGRADENHENCE_H

#include <cv.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace cv;
using namespace std;

typedef double ty;
typedef uchar tc;
#define Pi 3.14159265

class PMgradenhence{

public:
    int X_image,Y_image;
    int Tchannel,row,col,perPixel,maxPixel;

    void HistNormolize(Mat&pImg, Mat&pNormImg);

    void pmgrad(Mat&A,Mat&A1,Mat&B,double k,double dt);
    void defineChar(Mat&A);

    void diffusion(Mat &A, Mat &B,double max,double min);

    void gradn(Mat &A,Mat &B,double*max,double*min);
    void grads(Mat &A,Mat &B,double*max,double*min);
    void gradw(Mat &A,Mat &B,double*max,double*min);
    void grade(Mat &A,Mat &B,double*max,double*min);
    double Entropy(Mat &A);
    void MeanStdEntropy(Mat &A,string name);


};

#endif //PMGRADENHENCE_PMGRADENHENCE_H
