//
// Created by sunguoyan on 16/4/7.
//

#include "TVenhence.h"

void TVenhence::TVDenoising(Mat &src,Mat&dst, int n) {
    Mat src_0, src_x, src_xx, src_y, src_yy, src_xy, src_tmp1, src_tmp2, src_xy1, src_xy2, src_g, src_g3;
    src_0.create(src.size(), src.type());
    src_xx.create(src.size(), src.type());
    src_x.create(src.size(), src.type());
    src_y.create(src.size(), src.type());
    src_yy.create(src.size(), src.type());
    src_xy.create(src.size(), src.type());
    src_tmp1.create(src.size(), src.type());
    src_tmp2.create(src.size(), src.type());
    src_xy1.create(src.size(), src.type());
    src_xy2.create(src.size(), src.type());
    src_g.create(src.size(), src.type());
    src_g3.create(src.size(), src.type());
//    dst.create(src.size(), src.type());
//    dst = src;
    int b=1;
    int be = b*b;
    double dt = (double)b/5.0f;
    double l=0.0;

    int h, w, k, i;
    for (k = 0; k < n; k++) {
        //计算src_x,src_xx
        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image - 1; w++) {
                src_tmp1.at<ty>(h, w) = src.at<ty>(h, w + 1);
                src_tmp2.at<ty>(h, w + 1) = src.at<ty>(h, w);
            }
        }
        for (i = 0; i < Y_image; i++) {
            src_tmp1.at<ty>(i, X_image - 1) = src.at<ty>(i, X_image - 1);
            src_tmp2.at<ty>(i, 0) = src.at<ty>(i, 0);
        }

        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                src_x.at<ty>(h, w) = (src_tmp1.at<ty>(h, w) - src_tmp2.at<ty>(h, w)) / 2;
                src_xx.at<ty>(h, w) = src_tmp1.at<ty>(h, w) + src_tmp2.at<ty>(h, w) - 2 * src.at<ty>(h, w);
            }
        }

        //计算src_y,src_yy

        for (h = 0; h < Y_image - 1; h++) {
            for (w = 0; w < X_image; w++) {
                src_tmp1.at<ty>(h, w) = src.at<ty>(h + 1, w);
                src_tmp2.at<ty>(h + 1, w) = src.at<ty>(h, w);
            }
        }

        for (i = 0; i < X_image; i++) {
            src_tmp1.at<ty>(Y_image - 1, i) = src.at<ty>(Y_image - 1, i);
            src_tmp2.at<ty>(0, i) = src.at<ty>(0, i);
        }

        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                src_y = (src_tmp1.at<ty>(h, w) - src_tmp2.at<ty>(h, w)) / 2;
                src_yy = src_tmp1.at<ty>(h, w) + src_tmp2.at<ty>(h, w) - 2 * src.at<ty>(h, w);
            }
        }

        //计算src_xy
        for (h = 0; h < Y_image - 1; h++) {
            for (w = 0; w < X_image - 1; w++) {
                src_tmp1.at<ty>(h, w) = src.at<ty>(h + 1, w + 1);
                src_tmp2.at<ty>(h + 1, w + 1) = src.at<ty>(h, w);
            }
        }
        for (i = 0; i < Y_image - 1; i++) {
            src_tmp1.at<ty>(i, X_image - 1) = src.at<ty>(i + 1, X_image - 1);
            src_tmp2.at<ty>(i + 1, 0) = src.at<ty>(i, 0);
        }
        for (i = 0; i < X_image - 1; i++) {
            src_tmp1.at<ty>(Y_image - 1, i) = src.at<ty>(Y_image - 1, i + 1);
            src_tmp2.at<ty>(0, i + 1) = src.at<ty>(0, i);
        }
        src_tmp1.at<ty>(Y_image - 1, X_image - 1) = src.at<ty>(Y_image - 1, X_image - 1);
        src_tmp2.at<ty>(0, 0) = src.at<ty>(0, 0);

        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                src_xy1.at<ty>(h, w) = src_tmp1.at<ty>(h, w) - src_tmp2.at<ty>(h, w);
            }
        }

        for (h = 0; h < Y_image - 1; h++) {
            for (w = 0; w < X_image - 1; w++) {
                src_tmp1.at<ty>(h + 1, w) = src.at<ty>(h, w + 1);
                src_tmp2.at<ty>(h, w + 1) = src.at<ty>(h + 1, w);
            }
        }

        for (i = 0; i < Y_image - 1; i++) {
            src_tmp1.at<ty>(i + 1, X_image - 1) = src.at<ty>(i, X_image - 1);
            src_tmp2.at<ty>(i, 0) = src.at<ty>(i + 1, 0);
        }
        for (i = 0; i < X_image - 1; i++) {
            src_tmp1.at<ty>(0, i) = src.at<ty>(0, i + 1);
            src_tmp2.at<ty>(Y_image - 1, i + 1) = src.at<ty>(Y_image - 1, i);
        }
        src_tmp1.at<ty>(0, X_image - 1) = src.at<ty>(0, X_image - 1);
        src_tmp2.at<ty>(Y_image - 1, 0) = src.at<ty>(Y_image, 0);

        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                src_xy2.at<ty>(h, w) = src_tmp1.at<ty>(h, w) + src_tmp2.at<ty>(h, w);
            }
        }
        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                src_xy.at<ty>(h, w) = (src_xy1.at<ty>(h,w)-src_xy2.at<ty>(h,w))/4;
            }
        }
//         cout<<"xy"<<endl;

        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                src_g.at<ty>(h, w) = src_xx.at<ty>(h, w) * (src_y.at<ty>(h, w) * src_y.at<ty>(h, w) + be) -
                                     2 * src_x.at<ty>(h, w) * src_y.at<ty>(h, w) * src_xy.at<ty>(h, w) +
                                     src_yy.at<ty>(h, w) * (src_x.at<ty>(h, w) * src_y.at<ty>(h, w) + be);
                src_g3.at<ty>(h, w) = pow(
                        src_x.at<ty>(h, w) * src_x.at<ty>(h, w) + src_y.at<ty>(h, w) * src_y.at<ty>(h, w) + be, 1.5);
            }
        }

        for (h = 0; h < Y_image; h++) {
            for (w = 0; w < X_image; w++) {
                dst.at<ty>(h, w) = dst.at<ty>(h, w) + dt * (src_g.at<ty>(h, w) / src_g3.at<ty>(h, w) +
                                                            l * (src.at<ty>(h, w) - dst.at<ty>(h, w)));
                if(dst.at<ty>(h, w)>1){
                    dst.at<ty>(h, w) = 1;
                }
                if(dst.at<ty>(h, w)<0){
                    dst.at<ty>(h, w) = 0;
                }
            }
        }
    }

    return;
}




int main() {
    string filename = "/Users/sunguoyan/Downloads/picture/lenazao.bmp";
    Mat src,src1;
    src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src1,CV_32F,1/255.0);
    Mat dst = src1.clone();
    TVenhence p;
    p.X_image = src.cols;
    p.Y_image = src.rows;
    p.TVDenoising(src1,dst,20);
    namedWindow("orig");
    imshow("orig", src1);
    namedWindow("result");
    imshow("result", dst);
    waitKey(0);
    return 0;
}