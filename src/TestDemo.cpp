#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include "headers.h"
#include "IntImage.h"
#include "SimpleClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"
#include "Global.h"
#include <omp.h>

const string TriName = "./data/face.tri";
const string  ConName= "./data/face.con";
CvScalar color = CV_RGB(0,255,0);

void LoadTriPosition(Mat& Tri,string filename)
{
	fstream file(filename,fstream::in);
	if (!file.is_open())
	{
		cout<<"Failed to open file for reading\n";
		return ;
	}
	char str[256];
	char c;
	while(1)
	{
		file>>str;
		if (strncmp(str,"n_tri:",6) == 0)
			break;
	}
	int n ;
	file >> n;
	while(1)
	{
		file >> c; 
		if(c == '{')
			break;
	}
	for (int i = 0; i < n; i++)
	{
		file >> Tri.at<int>(i,0) >> Tri.at<int>(i,1) >> Tri.at<int>(i,2);
	}
	file.close();
}

void LoadCon(Mat&Con,string filename)
{
	int i,n;
	char str[256];
	char c; 
	fstream file(filename,fstream::in);
	if(!file.is_open())
	{
	cout<<"Failed to open file for reading\n";
		return ;
	}
	while(1)
	{
		file >> str; 
		if(strncmp(str,"n_connections:",14) == 0)
			break;
	}
	file >> n; 
	while(1)
	{
		file >> c; 
		if(c == '{')
			break;
	}
	for(i = 0; i < n; i++)
		file >> Con.at<int>(0,i) >> Con.at<int>(1,i);
	file.close();

}

void Draw(Mat image,Mat_<double>& shape,Mat Tri,Mat Con)
{
	int i,n = 68;
	cv::Point p1,p2;
	cv::Scalar c;

	//画出三角链接
	c = CV_RGB(0,0,0);
	for(i = 0; i < Tri.rows; i++)
	{
		p1 = cv::Point(shape.at<double>(Tri.at<int>(i,0),0),
				shape.at<double>(Tri.at<int>(i,0),1));
		p2 = cv::Point(shape.at<double>(Tri.at<int>(i,1),0),
				shape.at<double>(Tri.at<int>(i,1),1));
		cv::line(image,p1,p2,c);
		p1 = cv::Point(shape.at<double>(Tri.at<int>(i,0),0),
				shape.at<double>(Tri.at<int>(i,0),1));
		p2 = cv::Point(shape.at<double>(Tri.at<int>(i,2),0),
				shape.at<double>(Tri.at<int>(i,2),1));
		cv::line(image,p1,p2,c);
		p1 = cv::Point(shape.at<double>(Tri.at<int>(i,2),0),
				shape.at<double>(Tri.at<int>(i,2),1));
		p2 = cv::Point(shape.at<double>(Tri.at<int>(i,1),0),
				shape.at<double>(Tri.at<int>(i,1),1));
		cv::line(image,p1,p2,c);
	}
	//画出点链接区域
	c = CV_RGB(0,0,255);
	for(i = 0; i < Con.cols; i++)
	{
		p1 = cv::Point(shape.at<double>(Con.at<int>(0,i),0),
			shape.at<double>(Con.at<int>(0,i),1));
		p2 = cv::Point(shape.at<double>(Con.at<int>(1,i),0),
			shape.at<double>(Con.at<int>(1,i),1));
		cv::line(image,p1,p2,c,1);
	}

	CvFont font;
	cvInitFont(&font,CV_FONT_VECTOR0,1.5,0.8,0.0,1);
	//draw points
	c = CV_RGB(255,0,0);
	for(i = 0; i < n; i++)
	{
		p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i,1));
		cv::circle(image,p1,2,c);
	}
}

void facesDetect(Mat& img,cv::vector<Rect>&Crect)
{
	IntImage Intimg;
	Intimg.Load(img);
	cascade->ApplyOriginalSize(Intimg,Crect);
}


/*************************************************************************
		函数名称：warpImage
		函数功能：图像仿射变换
		函数参数：rc,landmark,
		函数输出：roi ,rotation_landmark
		创建时间:2016年9月14日 15:32
		创建人:张峰
		备注：
*************************************************************************/
Mat_<double> warpImage(Mat& src,Mat_<double>landmark,Mat& roi)
{
	Mat_<double>rotation_landmark(68,2,0.0);
	Mat temp;
	if (src.channels()==3)
	{
		cvtColor(src,temp,CV_BGR2GRAY);
	}
	else
		temp = src.clone();
	//人眼定位
	CvPoint leftEye = cvPoint(0,0);
	CvPoint rightEye = leftEye;
	CvPoint centerEye = leftEye;
	for (int i = 36; i < 42; i++)
	{
		leftEye.x+= landmark(i,0);
		leftEye.y+= landmark(i,1);
	}
	leftEye.x = cvRound(leftEye.x/6.0);
	leftEye.y =cvRound(leftEye.y/6.0);
	for (int i = 42; i <48; i++)
	{
		rightEye.x+=landmark(i,0);
		rightEye.y+=landmark(i,1);
	}
	rightEye.x = cvRound(rightEye.x/6.0);
	rightEye.y = cvRound(rightEye.y/6.0);
	
	centerEye.x = 0.5*(leftEye.x+rightEye.x);
	centerEye.y = 0.5*(leftEye.y+rightEye.y);

	//dy为正时头像左偏，顺时针旋转angle调整图像
	double dy = rightEye.y - leftEye.y;
	double dx = rightEye.x - leftEye.x;
	double len = sqrtf(dx*dx+dy*dy);
	double angle = atan2f(dy,dx)*180/CV_PI;

	//确定旋转后的图像大小
	Mat roi_mat = getRotationMatrix2D(centerEye,angle,1.0);

	Mat s = temp;
	warpAffine(s,roi,roi_mat,cvSize(temp.cols,temp.rows));

	//vector<Point2f> marks;
	//按照仿射变换矩阵，计算变换后各关键点在新图中所对应的位置坐标。
	for (int n = 0; n < landmark.rows; n++)
	{
		rotation_landmark(n,0) = roi_mat.ptr<double>(0)[0] *landmark(n,0) + roi_mat.ptr<double>(0)[1] * landmark(n,1) + roi_mat.ptr<double>(0)[2];
		rotation_landmark(n,1) = roi_mat.ptr<double>(1)[0] *landmark(n,0) + roi_mat.ptr<double>(1)[1] * landmark(n,1) + roi_mat.ptr<double>(1)[2];
	}
	return rotation_landmark;
}





void getFaceLandmark(const char* Cas_file_name,string imglist)
{
	//读取特征连线点坐标
	Mat Tri(90,3,CV_32S);
	LoadTriPosition(Tri,TriName);

	//读取三角形连线
	Mat Con(2,63,CV_32S);
	LoadCon(Con,ConName);

	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_DUPLEX,1.0f,1.0f,0,1,CV_AA);

	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(Cas_file_name);
	
	string image_file_name;
	fstream fin;
	fin.open(imglist,ios_base::in);
	while (fin>>image_file_name)
	{
		Mat img,gray;
		img = imread(image_file_name);
		if (img.channels()==3)
		{
			cvtColor(img,gray,CV_BGR2GRAY);
		}
		else
			gray = img.clone();
		Mat small_gray(gray.rows*0.5,gray.cols*0.5,gray.type());
		resize(gray, small_gray, small_gray.size(), 0, 0, INTER_AREA);
		vector<Rect>faces;
		facesDetect(small_gray,faces);
		Mat_<uchar>pridict_input_temp = gray.clone();
		for (int i = 0; i < faces.size(); i++)
		{
			BoundingBox bbox;
			bbox.start_x = 2*faces[i].x;
			bbox.start_y = 2*faces[i].y;
			bbox.width = 2*faces[i].width;
			bbox.height = 2*faces[i].height;
			bbox.center_x = bbox.start_x + bbox.width / 2.0;
			bbox.center_y = bbox.start_y + bbox.height / 2.0;
			cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bbox);
			Mat_<double>landmark = cas_load.Predict(pridict_input_temp,current_shape,bbox);
			Mat rotation_image;//旋转后的图像
			//rotation_landmark 为旋转后的68个特征点坐标
			Mat_<double>rotation_landmark = warpImage(img,landmark,rotation_image);
			rectangle(img,Point(bbox.start_x,bbox.start_y),Point(bbox.start_x+bbox.width,bbox.start_y+bbox.height),CV_RGB(255,0,0));
			Draw(img,landmark,Tri,Con);
			imshow("test",img);
			/*
			rectangle(rotation_image,Point(bbox.start_x,bbox.start_y),Point(bbox.start_x+bbox.width,bbox.start_y+bbox.height),CV_RGB(255,0,0));
			Draw(rotation_image,rotation_landmark,Tri,Con);
			imshow("test2",rotation_image);
			*/
			cvWaitKey(0);
		}
	}
}
int main()
{
	InitGlobalData();
	string cas_file_path = "F:\\Backup\\人脸检测与3000fps模型\\helen_trained_model\\helenModel";
	string img_file = "imglist.txt";
	getFaceLandmark(cas_file_path.c_str(),img_file);
	return 0;
}