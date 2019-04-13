#define _SCL_SECURE_NO_WARNINGS
#include"NN.h"
#include<iostream>

int main() {

	int trainnum = 100;
	int samplenum = 200;
	int classnum = 5;
	
	cv::Mat trainData, trainLabel, testData, testLabel;

	cv::Mat featureMat;
	std::string filepath = "D:\\C++\\ML\\ML\\Res\\" + std::to_string(1) + ".xml";
	cv::FileStorage fs(filepath, cv::FileStorage::READ);
	fs["featureMat"] >> featureMat;
	fs.release();

	int featureDim = featureMat.cols;
	int testnum = samplenum - trainnum;

	trainData.create(trainnum*classnum, featureDim, CV_32FC1);
	trainLabel.create(trainnum*classnum, 1, CV_32FC1);
	testData.create(testnum*classnum, featureDim, CV_32FC1);
	testLabel.create(testnum*classnum, 1, CV_32FC1);

	for (int i = 0; i<classnum; i++) {
		std::string fsname = "D:\\C++\\ML\\ML\\Res\\" + std::to_string(i + 1) + ".xml";
		featureMat.release();
		fs.open(fsname, cv::FileStorage::READ);
		fs["featureMat"] >> featureMat;
		fs.release();

		assert(featureDim == featureMat.cols);

		std::copy(featureMat.begin<float>(), featureMat.begin<float>() + featureDim * trainnum, trainData.begin<float>() + i * trainnum*featureDim);
		std::copy(featureMat.begin<float>() + featureDim * trainnum, featureMat.end<float>(), testData.begin<float>() + i * testnum*featureDim);

		for (int j = i * trainnum; j<(i + 1)*trainnum; j++) {
			trainLabel.at<float>(j, 0) = i;
		}

		for (int j = i * testnum; j < (i + 1)*testnum; j++) {
			testLabel.at<float>(j, 0) = i;
		}
		std::cout << i << std::endl;
		featureMat.release();
	}
	std::cout << "Divede Successfully" << std::endl;

	Parameters para;
	para.iteration = 10000;
	para.alpha = 0.007;
	para.classify_num = classnum;

	std::vector<int> NN_size{500,50,classnum};

	NN nn(NN_size, para);
	nn.train(trainData,trainLabel);
	nn.predict(testData, testLabel);
	return 1;
}