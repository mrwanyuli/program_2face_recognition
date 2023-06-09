#include <iostream>
#include "opencv2/opencv.hpp"
#include "face.h"

using namespace std;
using namespace cv;
using namespace aip;

int main()
{
    VideoCapture cap(0); 
    if( !cap.isOpened() )
    {
	cout << "Camera open failed" << endl;
	return 0;
    }
    cout << "Camera open success" << endl;

    CascadeClassifier Classifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml");

    aip::Face client("你的AppID", "你的API Key", "你的Secret Key");

    Mat ColorImage;
    Mat GrayImage;
    vector<Rect> AllFace;
    Mat MatFace;
    vector<uchar> JpgFace;
    string Base64Face;
    Json::Value result;
    time_t sec;
   
    for(;;)
    {
	cap >> ColorImage;
	cvtColor(ColorImage, GrayImage, CV_BGR2GRAY);
	equalizeHist(GrayImage, GrayImage);
	Classifier.detectMultiScale(GrayImage, AllFace);
	if( AllFace.size() )
	{	
	    rectangle(GrayImage, AllFace[0], Scalar(255,255,255));
	    MatFace = GrayImage(AllFace[0]);
	    imencode(".jpg", MatFace, JpgFace); 
	    Base64Face = base64_encode((char *)JpgFace.data(), JpgFace.size());
	    result = client.search(Base64Face, "BASE64", "Teaching", aip::null);
	    if( !result["result"].isNull() )
	    {
		if( result["result"]["user_list"][0]["score"].asInt() > 80 )
		{
		    cout << result["result"]["user_list"][0]["user_id"] << endl; 
		    sec = time(NULL);
		    cout << ctime(&sec) << endl;
		    putText(GrayImage, result["result"]["user_list"][0]["user_id"].asString(), Point(0,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255));
		    putText(GrayImage, ctime(&sec), Point(0,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255));
		}
	    }
	}
	imshow("video", GrayImage);
	waitKey(40);
    }

    return 0;
}

