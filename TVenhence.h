//
// Created by sunguoyan on 16/4/7.
//

#ifndef PMGRADENHENCE_TVENHENCE_H
#define PMGRADENHENCE_TVENHENCE_H
#include<iostream>
#include<opencv2/highgui.hpp>
#include<cv.h>
typedef float ty;
typedef char ti;
using namespace std;
using namespace cv;

class TVenhence{
public:
    int X_image,Y_image;
    void TVDenoising(Mat&src,Mat&dst,int n);

};

#endif //PMGRADENHENCE_TVENHENCE_H
