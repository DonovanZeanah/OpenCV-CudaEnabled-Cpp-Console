#include "pch.h"

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>





using namespace std;
using namespace cv;
using namespace dnn;
//using namespace cuda; //commented for clear seperation in codee

VideoCapture cap(0);

int main(int argc, char** argv) {

		cuda::printCudaDeviceInfo(0);

		Mat img;
		cuda:: GpuMat imgGpu;

		while (cap.isOpened()){

				auto start = getTickCount();
				cap.read(img);
				imgGpu.upload(img);

				
				cuda::cvtColor(imgGpu, imgGpu, COLOR_BGR2GRAY);

				// Image Filtering
				//auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, {3,3}, 1);
				//gaussianFilter->apply(imgGpu, imgGpu);

				//auto laplacianFilter = cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 3,3);
				//laplacianFilter->apply(imgGpu, imgGpu);

				//auto morpFilter = cuda::createMorphologyFilter(MORPH_CLOSE, CV_8UC1, getStructuringElement(MORPH_RECT, { 3, 3 }));
				//morpFilter->apply(imgGpu, imgGpu);



				imgGpu.download(img);

				auto end = getTickCount();
				auto totalTime = (end - start) / getTickFrequency();
				auto fps = 1 / totalTime;

				putText(img, "FPS: " + to_string(int(fps)), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 120), 2, 8, false);

				//imshow("img", img);
				imshow("Image", img);
				if (waitKey(1) == 'q') {
						break;
				}
						
		}
}
