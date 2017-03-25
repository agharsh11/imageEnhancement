#include<iostream>
#include<opencv/highgui.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<stdlib.h>
//#include "opencv2/videoio.hpp"
using namespace std;
using namespace cv;

const double lambda = 1.0 / 7.0;
const double k = 30;
const int iter = 5;


float ahN[3][3] = { {0, 1, 0}, {0, -1, 0}, {0, 0, 0} };
float ahS[3][3] = { {0, 0, 0}, {0, -1, 0}, {0, 1, 0} };
float ahE[3][3] = { {0, 0, 0}, {0, -1, 1}, {0, 0, 0} };
float ahW[3][3] = { {0, 0, 0}, {1, -1, 0}, {0, 0, 0} };
// float ahNE[3][3] = { {0, 0, 1}, {0, -1, 0}, {0, 0, 0} };
// float ahSE[3][3] = { {0, 0, 0}, {0, -1, 0}, {0, 0, 1} };
// float ahSW[3][3] = { {0, 0, 0}, {0, -1, 0}, {1, 0, 0} };
// float ahNW[3][3] = { {1, 0, 0}, {0, -1, 0}, {0, 0, 0} };

Mat hN = Mat(3, 3, CV_32FC1, &ahN);
Mat hS = Mat(3, 3, CV_32FC1, &ahS);
Mat hE = Mat(3, 3, CV_32FC1, &ahE);
Mat hW = Mat(3, 3, CV_32FC1, &ahW);
// Mat hNE = Mat(3, 3, CV_32FC1, &ahNE);
// Mat hSE = Mat(3, 3, CV_32FC1, &ahSE);
// Mat hSW = Mat(3, 3, CV_32FC1, &ahSW);
// Mat hNW = Mat(3, 3, CV_32FC1, &ahNW);


void anisotropicDiffusion(Mat &output, int width, int height) 
{	
	//mat initialisation
	Mat nablaN, nablaS, nablaW, nablaE, nablaNE, nablaSE, nablaSW, nablaNW;
	Mat cN, cS, cW, cE, cNE, cSE, cSW, cNW;

	//depth of filters
	int ddepth = -1;

	//center pixel distance
	double dx = 1, dy = 1, dd = sqrt(2);
	double idxSqr = 1.0 / (dx * dx), idySqr = 1.0 / (dy * dy), iddSqr = 1 / (dd * dd);
	
	for (int i = 0; i < iter; i++) {
		
		//filters 
		filter2D(output, nablaN, ddepth, hN);
		filter2D(output, nablaS, ddepth, hS);
		filter2D(output, nablaW, ddepth, hW);
		filter2D(output, nablaE, ddepth, hE);
		// filter2D(output, nablaNE, ddepth, hNE);
		// filter2D(output, nablaSE, ddepth, hSE);
		// filter2D(output, nablaSW, ddepth, hSW);
		// filter2D(output, nablaNW, ddepth, hNW);

		//exponential flux
		cN = nablaN / k;
		cN = cN.mul(cN);
		cN = 1.0 / (1.0 + cN);
		//exp(-cN, cN);

		cS = nablaS / k;
		cS = cS.mul(cS);
		cS = 1.0 / (1.0 + cS);
		//exp(-cS, cS);

		cW = nablaW / k;
		cW = cW.mul(cW);
		cW = 1.0 / (1.0 + cW);
		//exp(-cW, cW);

		cE = nablaE / k;
		cE = cE.mul(cE);
		cE = 1.0 / (1.0 + cE);
		//exp(-cE, cE);

		// cNE = nablaNE / k;
		// cNE = cNE.mul(cNE);
		// cNE = 1.0 / (1.0 + cNE);
		// //exp(-cNE, cNE);

		// cSE = nablaSE / k;
		// cSE = cSE.mul(cSE);
		// cSE = 1.0 / (1.0 + cSE);
		// //exp(-cSE, cSE);

		// cSW = nablaSW / k;
		// cSW = cSW.mul(cSW);
		// cSW = 1.0 / (1.0 + cSW);
		// //exp(-cSW, cSW);

		// cNW = nablaNW / k;
		// cNW = cNW.mul(cNW);
		// cNW = 1.0 / (1.0 + cNW);
		// //exp(-cNW, cNW);

		output = output + lambda * (idySqr * cN.mul(nablaN) + idySqr * cS.mul(nablaS) + 
									idxSqr * cW.mul(nablaW) + idxSqr * cE.mul(nablaE) /*+
									iddSqr * cNE.mul(nablaNE) + iddSqr * cSE.mul(nablaSE) +
									iddSqr * cNW.mul(nablaNW) + iddSqr * cSW.mul(nablaSW)*/);
	}
}
int main()
{
	Mat src;
	Mat input1;
	VideoCapture cap(1);
       	if( !cap.isOpened() )
        throw "Error when reading steam_avi";

    	namedWindow("w", 1);
    	for( ; ; )
    	{
        	cap >> src;
            cv::resize(src, src, Size(600,600));
        	cv::Mat input[3],lab_image,tmp_plane,temp;
       		src.copyTo(input1);
       		imshow("dsigher",input1);
       		cv::split(input1, input);
       		//cv::cvtColor(input[0], lab_image, CV_BGR2Lab);
       		int width = input1.cols;
			int height = input1.rows;
			vector<Mat> out;
			//out.push_back();
			temp = input[0].clone();
			out.push_back(temp);
			temp = input[1].clone();
			out.push_back(temp);
			temp = input[2].clone();
			out.push_back(temp);
			out[0].convertTo(out[0], CV_32FC1);
			out[1].convertTo(out[1], CV_32FC1);
			out[2].convertTo(out[2], CV_32FC1);
			anisotropicDiffusion(out[0], width, height);
			//anisotropicDiffusion(out[1], width, height);
			//anisotropicDiffusion(out[2], width, height);
			double min[3];
			double max[3];
			minMaxIdx(out[0], &min[0], &max[0]);
			//minMaxIdx(out[1], &min[1], &max[1]);
			//minMaxIdx(out[2], &min[2], &max[2]);
			out[0].convertTo(out[0], CV_8UC1, 255 / (max[0]- min[0]), -min[0]);
			out[1].convertTo(out[1], CV_8UC1/*, 255 / (max[1]- min[1]), -min[1]*/);
			out[2].convertTo(out[2], CV_8UC1/*, 255 / (max[2]- min[2]), -min[2]*/);
			
       		// apply the CLAHE algorithm to the L channel
       		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
       		clahe->setClipLimit(4);
       		cv::Mat dst;
       		clahe->apply(out[0], out[0]);
       		clahe->apply(out[1], out[1]);
            clahe->apply(out[2], out[2]);
            
            cv::merge(out, input1);
            
            Mat input2[3];
            
            imshow("higher",input1);
            cv::cvtColor(input1, input1, CV_BGR2YCrCb);
       		//cv::cvtColor(input[0], lab_image, CV_BGR2Lab);
       		cv::split(input1, input2);
       		vector<Mat> out1;
       		temp = input2[0].clone();
			out1.push_back(temp);
			temp = input2[1].clone();
			out1.push_back(temp);
			temp = input2[2].clone();
			out1.push_back(temp);
			cv::Ptr<cv::CLAHE> clahe2 = cv::createCLAHE();
			clahe2->setClipLimit(4);
          	clahe2->apply(out1[0], out1[0]);
			cv::merge(out1, input1);
			Mat dstr; 
			cv::cvtColor(input1, dstr, CV_YCrCb2BGR);
       		//out.push_back();
			/*temp = input[0].clone();
			out.push_back(temp);
			temp = input[1].clone();
			out.push_back(temp);
			temp = input[2].clone();
			out.push_back(temp);
			cv::Ptr<cv::CLAHE> clahe2 = cv::createCLAHE();
			clahe2->setClipLimit(4);
          	clahe2->apply(out[0], out[0]);

          	out[0].convertTo(out[0], CV_32FC1);
			//out[1].convertTo(out[1], CV_32FC1);
			//out[2].convertTo(out[2], CV_32FC1);
            cv::merge(out, input1);
            Mat dstr; 
            cv::cvtColor(input1, dstr, CV_YCrCb2BGR);
          */
          // adaptive threshold
          
          /*CvRect roi;                                                             //Rectangular ROI
          CvSize size;
          int r, c, N=20;
          int count = 0;                                               
          CvScalar meanValue;      
          Mat imgDummy;
          imgDummy=lab_planes[0];                                               
          size = cvGetSize(imgDummy);                                            
          for (r = 9; r < 590; r ++)
          {
            for (c = 9; c < 590; c ++)
            {
              count++;
              roi.x = c-9;
              roi.y = r-9;
              roi.width = 20;
              roi.height = 20;
              cvSetImageROI(imgDummy, roi);
              meanValue = cvAvg(imgDummy);
              if(meanValue<imgDummy[r][c])
              {
                  cout<<"yes"<<endl;cv
              }
              else
                cout<<"NO"<<endl;
              cvResetImageROI(imgDummy);
            }
          } 
       		
		      // Merge the the color planes back into an Lab image
       		dst.copyTo(lab_planes[0]);
       		cv::merge(lab_planes, lab_image);

      		// convert back to RGB
      		//cv::cvtColor(lab_image, InImage, CV_Lab2BGR);*/

      		imshow("luma_corrected",dstr);
      		//waitKey(0);;
        	imshow("w", src);
        	waitKey(1); // waits to display frame
    	}
    	waitKey(0);
}
