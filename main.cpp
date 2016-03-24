#include "PMgradenhence.h"

typedef double ty;
typedef uchar tc;
#define Pi 3.14159265

//求最大梯度模与最小梯度模
void PMgradenhence::max_min_grad(Mat&A,double*max,double*min){
    int h,w;
    double ux,uy,u;
    *max = 0;
    *min = 255;
    for(h = 0;h < Y_image - 1;h++){
        for(w = 0;w < X_image - 1;w++){
            ux = A.at<ty>(h,w+1) - A.at<ty>(h,w);
            uy = A.at<ty>(h + 1,w) - A.at<ty>(h,w);
            u = sqrt(ux*ux+uy*uy);
            if(u > *max){
                *max = u;
            }
            if(u <  *min){
                *min = u;
            }
        }
    }
    return;
}


//将图像转换到梯度场，忽略了后面的方向处理的
void PMgradenhence::gradenhence(Mat&A,Mat&B){
    int h ,w;
    double gmax,gmin,ux,uy,u;
    max_min_grad(A,&gmax,&gmin);
    for(h = 0;h < Y_image - 1;h++){
        for(w = 0; w < X_image - 1;w++){
            ux = A.at<ty>(h,w+1) - A.at<ty>(h,w);
            uy = A.at<ty>(h + 1,w) - A.at<ty>(h,w);
            u = sqrt(ux*ux+uy*uy);
            B.at<ty>(h,w) = (1-cos(((u-gmin)/(gmax-gmin))*Pi))*(gmax/2);
        }
    }

}


void PMgradenhence::HistNormolize(Mat&pImg, Mat&pNormImg)
{
    int hist[256];
    double  fpHist[256];
    double eqHistTemp[256];
    int eqHist[256];
    int size = X_image *Y_image;
    int i ,j;
    memset(&hist,0,sizeof(int)*256);
    memset(&fpHist,0,sizeof(double)*256);
    memset(&eqHistTemp,0,sizeof(double)*256);
    for (i = 0;i < Y_image; i++) //计算差分矩阵直方图
    {
        for (j = 0; j < X_image; j++)
        {
            unsigned char GrayIndex = pImg.at<tc>(i,j);
            hist[GrayIndex] ++ ;
        }
    }
    for (i = 0; i< 256; i++)   // 计算灰度分布密度
    {
        fpHist[i] = (double)hist[i] / (double)size;
    }
    for ( i = 1; i< 256; i++)   // 计算累计直方图分布
    {
        if (i == 0)
        {
            eqHistTemp[i] = fpHist[i];
        }
        else
        {
            eqHistTemp[i] = eqHistTemp[i-1] + fpHist[i];
        }
    }
    //累计分布取整，保存计算出来的灰度映射关系
    for (i = 0; i< 256; i++)
    {
        eqHist[i] = (int)(255.0 * eqHistTemp[i] + 0.5);
    }
    for (i = 0;i < Y_image; i++) //进行灰度映射 均衡化
    {
        for (j = 0; j < X_image; j++)
        {
            unsigned char GrayIndex = pImg.at<tc>(i,j);
            pNormImg.at<tc>(i,j) = eqHist[GrayIndex];
        }
    }
}
void PMgradenhence::gradu(Mat&A,Mat&B){
    int h,w;
    for(h = 1;h<Y_image;h++){
        for(w = 0;w < X_image;w++){
            B.at<ty>(h,w) = A.at<ty>(h,w) - A.at<ty>(h-1,w);
        }
    }
}

void PMgradenhence::gradz(Mat&A,Mat&B){
    int h,w;
    for(h = 0;h<Y_image;h++){
        for(w = 1;w < X_image;w++){
            B.at<ty>(h,w) = A.at<ty>(h,w) - A.at<ty>(h,w-1);
        }
    }
}

void PMgradenhence::converttochar(Mat&A,Mat&B){
    int h,w;
    for(h = 0;h < Y_image;h++){
        for(w = 0;w < X_image;w++){
            B.at<tc>(h,w) =(tc)( A.at<ty>(h,w)*255);
        }
    }
}



void PMgradenhence::pmgrad(Mat&src,Mat&src1,Mat&dst,double k){
    int h,w;
    Mat S,H,Gradz,Gradu,H1;
    H.create(src1.size(),src1.type());
    H1.create(src.size(),src.type());
    S.create(src.size(),src.type());
    Gradz.create(src.size(),src.type());
    Gradu.create(src.size(),src.type());

    gradenhence(src,S);
    HistNormolize(src1,H);


//    converttochar(H,H1);
    H.convertTo(H1,CV_64F,1.0/255.0,0);
//    cvConvertScaleAbs(&H,&H1);

//    namedWindow("zhifangtu");
//    imshow("zhifangtu",H1);
//    waitKey(6000);
    gradz(S,Gradz);
    gradu(S,Gradu);

    for(h = 0;h < Y_image;h++){
        for(w = 0;w < X_image;w++){
            dst.at<ty>(h,w) = dst.at<ty>(h,w)+((Gradu.at<ty>(h,w)+Gradz.at<ty>(h,w))+k*(H1.at<ty>(h,w)-src.at<ty>(h,w)));
        }
    }

    return;
}


void PMgradenhence::panduan(Mat&A){
    int h,w;
    for(h = 0;h < Y_image;h++){
        for(w = 0;w < X_image;w++){
            if(A.at<tc>(h,w)<90){
                cout<<"小于0！"<<endl;
                return;
            }
            if(A.at<tc>(h,w) > 200){
                cout<<"大于255！"<<endl;
                return;
            }
        }
    }
}

int main() {

    Mat src,src1,dst;
    PMgradenhence p;
    string filename = "/Users/sunguoyan/Downloads/picture/lena.bmp";
    src = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src1,CV_64F,1.0/255.0,0);
    p.Y_image = src.rows;
    p.X_image = src.cols;

    dst.create(src1.size(),src1.type());
    //测试转换到梯度场
    p.gradenhence(src1,dst);
    //测试直方图均衡化
//    p.HistNormolize(src,dst);
//    p.panduan(src);
//    cout<<"over"<<endl;
    //进行梯度场结合偏微分方程增强
    double k;
    k = 1;
//    p.pmgrad(src1,src,dst,k);

    namedWindow("test");
    imshow("test",src1);
    namedWindow("result");
    imshow("result",dst);
    waitKey(0);
    return 0;
}