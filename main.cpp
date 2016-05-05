#include "PMgradenhence.h"


void PMgradenhence::gradn(Mat &A, Mat &B, double *nmax, double *nmin)    //求N方向梯度
{
    double tmp;
    *nmax = 0;
    *nmin = 255;
    int h, w;
    for (h = 1; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
            tmp = A.at<ty>(h - 1, w) - A.at<ty>(h, w);
            B.at<ty>(h, w) = tmp;
            if (fabs(tmp) > *nmax) {
                *nmax = fabs(tmp);
            }
            if (fabs(tmp) < *nmin) {
                *nmin = fabs(tmp);
            }
        }
    cout << *nmax << "--" << *nmin << endl;
}

void PMgradenhence::grads(Mat &A, Mat &B, double *smax, double *smin)    //求S方向梯度
{
    double tmp;
    *smax = 0;
    *smin = 255;
    int h, w;
    for (h = 0; h < Y_image - 1; h++)
        for (w = 0; w < X_image; w++) {
            tmp = A.at<ty>(h + 1, w) - A.at<ty>(h, w);
            B.at<ty>(h, w) = tmp;
            if (fabs(tmp) > *smax) {
                *smax = fabs(tmp);
            }
            if (fabs(tmp) < *smin) {
                *smin = fabs(tmp);
            }
        }
    cout << *smax << "--" << *smin << endl;
}

void PMgradenhence::grade(Mat &A, Mat &B, double *emax, double *emin)    //求E方向梯度
{
    double tmp;
    *emax = 0;
    *emin = 255;
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image - 1; w++) {
            tmp = A.at<ty>(h, w + 1) - A.at<ty>(h, w);
            B.at<ty>(h, w) = tmp;
            if (fabs(tmp) > *emax) {
                *emax = fabs(tmp);
            }
            if (fabs(tmp) < *emin) {
                *emin = fabs(tmp);
            }
        }
    cout << *emax << "--" << *emin << endl;
}

void PMgradenhence::gradw(Mat &A, Mat &B, double *wmax, double *wmin)    //求W方向梯度
{
    double tmp;
    *wmax = 0;
    *wmin = 255;
    int h, w;
    for (h = 0; h < Y_image; h++)
        for (w = 1; w < X_image; w++) {
            tmp = A.at<ty>(h, w - 1) - A.at<ty>(h, w);
            B.at<ty>(h, w) = tmp;
            if (abs(tmp) > *wmax) {
                *wmax = fabs(tmp);
            }
            if (abs(tmp) < *wmin) {
                *wmin = fabs(tmp);
            }
        }
    cout << *wmax << "--" << *wmin << endl;
}



void PMgradenhence::diffusion(Mat &A, Mat &B, double max, double min) //求扩散系数
{
    int h, w;
    double k = 1;
    double tmp;
    for (h = 0; h < Y_image; h++)
        for (w = 0; w < X_image; w++) {
//            B.at<ty>(h, w) = (1 - cos(((fabs(A.at<ty>(h, w))*fabs(A.at<ty>(h, w)) - min) / max - min) * Pi)) * (max / (2.0 * fabs(A.at<ty>(h, w))*fabs(A.at<ty>(h, w))));
//                B.at<ty>(h, w) =
//                        (1 - cos(((fabs(A.at<ty>(h, w)) - min) / max - min) * Pi)) * (max / (2.0 ));
            //赵建论文中的扩散系数
//            tmp =
//                    (1 - cos((((A.at<ty>(h, w)) - min) / max - min) * Pi)) * (max / (2.0 * (A.at<ty>(h, w))));
            //结合前向扩散与后向扩散的系数
            tmp = 1 / (1 + ((A.at<ty>(h, w)) / 40) * ((A.at<ty>(h, w)) / 40) * ((A.at<ty>(h, w)) / 40) *
                           ((A.at<ty>(h, w)) / 40)) -
                  0.25 / (1 + (((A.at<ty>(h, w)) - 80) / 20) * (((A.at<ty>(h, w)) - 80) / 20));

//            if (tmp > 0) {
//                B.at<ty>(h, w) = tmp;
//            }
//            if (tmp == 0) {
//                B.at<ty>(h, w) = exp(-6);
////                B.at<ty>(h, w) = A.at<ty>(h,w);
//            }
            B.at<ty>(h, w) = tmp;



//

        }
}

//直方图均衡化
void PMgradenhence::HistNormolize(Mat &pImg, Mat &pNormImg) {
    int hist[256];
    double fpHist[256];
    double eqHistTemp[256];
    int eqHist[256];
    int size = X_image * Y_image;
    int i, j;
    memset(&hist, 0, sizeof(int) * 256);
    memset(&fpHist, 0, sizeof(double) * 256);
    memset(&eqHistTemp, 0, sizeof(double) * 256);
    for (i = 0; i < Y_image; i++) //计算差分矩阵直方图
    {
        for (j = 0; j < X_image; j++) {
            unsigned char GrayIndex = pImg.at<tc>(i, j);
            hist[GrayIndex]++;
        }
    }
    for (i = 0; i < 256; i++)   // 计算灰度分布密度
    {
        fpHist[i] = (double) hist[i] / (double) size;
    }
    for (i = 1; i < 256; i++)   // 计算累计直方图分布
    {
        if (i == 0) {
            eqHistTemp[i] = fpHist[i];
        }
        else {
            eqHistTemp[i] = eqHistTemp[i - 1] + fpHist[i];
        }
    }
    //累计分布取整，保存计算出来的灰度映射关系
    for (i = 0; i < 256; i++) {
        eqHist[i] = (int) (255.0 * eqHistTemp[i] + 0.5);
    }
    for (i = 0; i < Y_image; i++) //进行灰度映射 均衡化
    {
        for (j = 0; j < X_image; j++) {
            unsigned char GrayIndex = pImg.at<tc>(i, j);
            pNormImg.at<tc>(i, j) = eqHist[GrayIndex];
        }
    }
}


//计算图片的信息熵
double PMgradenhence::Entropy(Mat &A) {
    int En[256];
    double fpEn[256];
    double result, sum = 0, tmp;
    int size = X_image * Y_image;
    int i, j;
    memset(&En, 0, sizeof(int) * 256);
    memset(&fpEn, 0, sizeof(double) * 256);
    for (i = 0; i < Y_image; i++) //计算差分矩阵直方图
    {
        for (j = 0; j < X_image; j++) {
            tc GrayIndex = A.at<tc>(i, j);
            En[GrayIndex]++;
        }
    }

    for (i = 0; i < 256; i++)   // 计算灰度分布密度
    {
        fpEn[i] = (double) En[i] / (double) size;
    }
    for (i = 0; i < 256; i++) {
        if (fpEn[i] > 0) {
            tmp = log(fpEn[i]) / log(2);
            sum = sum + fpEn[i] * tmp;
        }
    }
    sum = -sum;
    return sum;
}

//实现基于梯度场的增强，第一个参数是浮点图像，第二个是uchar类型图像
void PMgradenhence::pmgrad(Mat &src, Mat &src1, Mat &dst, double k, double dt) {
    int h, w;
    Mat S, H, Gradz, Gradu, H1;


    double is = 0, in = 0, iw = 0, ie = 0, cn = 0, cs = 0, cw = 0, ce = 0, sum = 0;


    Mat IN, IS, IW, IE, CN, CS, CW, CE;
    H.create(src1.size(), src1.type());
    H1.create(src.size(), src.type());

    IN.create(src.size(), src.type());
    IS.create(src.size(), src.type());
    IW.create(src.size(), src.type());
    IE.create(src.size(), src.type());
    CN.create(src.size(), src.type());
    CS.create(src.size(), src.type());
    CW.create(src.size(), src.type());
    CE.create(src.size(), src.type());

    double nmax, nmin, smax, smin, emax, emin, wmax, wmin;

    gradn(src, IN, &nmax, &nmin);       //求N方向的梯度
    grads(src, IS, &smax, &smin);        //求S方向的梯度
    grade(src, IE, &emax, &emin);        //求E方向的梯度
    gradw(src, IW, &wmax, &wmin);        //求W方向的梯度

//    对原图像进行直方图均衡
    HistNormolize(src1, H);
    H.convertTo(H1, CV_64F, 1.0 / 255.0, 0);

//    namedWindow("zhifangtu");
//    imshow("zhifangtu",H1);
//    waitKey(6000);

//求系数(1-cos((||u||-min)/(max-min))*pi)*(max/(2*||u||))
    diffusion(IN, CN, nmax, nmin);
    diffusion(IS, CS, smax, smin);
    diffusion(IE, CE, emax, emin);
    diffusion(IW, CW, wmax, wmin);
    int n = 0;
    double tmp;
    for (n = 0; n < 1; n++) {
        for (h = 1; h < Y_image - 1; h++) {
            for (w = 1; w < X_image - 1; w++) {
                in = src.at<ty>(h - 1, w) - src.at<ty>(h, w);
                is = src.at<ty>(h + 1, w) - src.at<ty>(h, w);
                iw = src.at<ty>(h, w - 1) - src.at<ty>(h, w);
                ie = src.at<ty>(h, w + 1) - src.at<ty>(h, w);
//            dst.at<ty>(h,w) = dst.at<ty>(h,w)+((Gradu.at<ty>(h,w)+Gradz.at<ty>(h,w))+k*(H1.at<ty>(h,w)-src.at<ty>(h,w)));
                sum = in * CN.at<ty>(h, w) + is * CS.at<ty>(h, w) + iw * CW.at<ty>(h, w) +
                      ie * CE.at<ty>(h, w);
//                sum = iw * CW.at<ty>(h, w) + ie * CE.at<ty>(h, w);
                sum = sum * dt;
//                cout<<sum<<endl;
                tmp = dst.at<ty>(h, w) + sum + dt * k * (H1.at<ty>(h, w) - src.at<ty>(h, w));

                dst.at<ty>(h, w) = tmp;

//                else {
//                    src.at<ty>(h, w) = src.at<ty>(h, w);
//                }
            }
        }
    }


    return;
}


void PMgradenhence::defineChar(Mat &A) {
    int h, w;
    for (h = 0; h < Y_image; h++) {
        for (w = 0; w < X_image; w++) {
            if (A.at<tc>(h, w) < 0) {
                cout << "小雨0" << endl;
                return;
            }
            if (A.at<tc>(h, w) > 100) {
                cout << "daledale" << endl;
                return;
            }
        }
    }
}

//计算并输出图像的均值，标准差，信息熵
void PMgradenhence::MeanStdEntropy(Mat &A, string name) {
    //计算图像信息熵
    double entropy;
    entropy = Entropy(A);

    //计算图像方差与均值
    Scalar mean;
    Scalar stddev;

    meanStdDev(A, mean, stddev);
    ty mean_pxl = mean.val[0];
    ty stddev_pxl = stddev.val[0];
    cout << name << "均值：" << mean_pxl << " " + name << "的标准差：" << stddev_pxl << " " + name << "的信息熵：" << entropy <<
    endl;
    return;
}

int main() {

    Mat src, src1, Gdst, udist;
    PMgradenhence p;
    double segma;
    string filename = "/Users/sunguoyan/Downloads/picture/lenazao.bmp";

    src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src1, CV_64F, 1.0 / 255.0, 0);
    p.Y_image = src.rows;
    p.X_image = src.cols;

    Mat dst = src1.clone();
//    dst.create(src1.size(), src1.type());

//    测试转换到梯度场
//    p.gradenhence(src1,dst);
//    测试直方图均衡化
//    p.HistNormolize(src,dst);
//    p.panduan(src);
//    cout<<"over"<<endl;
//    进行梯度场结合偏微分方程增强
//    王翠翠论文说dt<=0.25较理想
    double k, dt;
    k = 3;
    dt = 0.2;
    clock_t start, finish;
    double totaltime;
    start = clock();

    p.pmgrad(src1, src, dst, k, dt);

    finish = clock();
    totaltime = (double) (finish - start) / CLOCKS_PER_SEC;
    cout << "pm增强程序的运行时间为" << totaltime << "s！" << endl;

    dst.convertTo(udist, CV_8U, 255, 0);


    p.MeanStdEntropy(src, "原图像");
    p.MeanStdEntropy(udist, "增强后图像");

    namedWindow("test");
    imshow("test", src);
    namedWindow("result");
    imshow("result", udist);
    waitKey(0);
    return 0;
}