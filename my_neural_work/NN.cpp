#include "NN.h"

void NN::train(cv::Mat trainData, cv::Mat trainLabel)
{
	assert(trainData.rows == trainLabel.rows);

	traindata = trainData;
	trainlabel = trainLabel;
	cv::Mat temp = RandMat(traindata.cols, NN_size.at(0));
	NN_w.push_back(temp);
	temp.release();
	for (int i = 0; i<NN_size.size()-1; ++i) {
		cv::Mat temp_w = RandMat(NN_size.at(i),NN_size.at(i+1));
		cv::Mat temp_b = RandMat(1, NN_size.at(i));
		NN_w.push_back(temp_w);
		NN_b.push_back(temp_b);
	}
	temp= RandMat(1, NN_size.at(NN_size.size() - 1));
	NN_b.push_back(temp);

	//样本打乱处理；
	Mat_Shuffle(traindata, trainlabel);
	//训练标签平铺；
	trainlabel = One_hot(trainlabel,para.classify_num);

	float loss = Loss();
	float loss_old = loss;
	for (int t = 0; t < para.iteration; ++t) {
		int mini_batch = (t*para.batch_size) % traindata.rows;
		//随机梯度下降，更新参数列表W;
		Mini_SGD(mini_batch);

		if (t % 10 == 0) {
			loss = Loss();
			if (loss < loss_old) {
				std::cout << "T= " << t << "  Loss= " << loss << "  down" << std::endl;
			}
			else {
				std::cout << "T= " << t << "  Loss= " << loss << "  up" << std::endl;
			}
			loss_old = loss;
		}
	}

	saveModel();
}

void NN::predict(cv::Mat testdata, cv::Mat testlabel)
{
	assert(testdata.rows == testlabel.rows);
	std::vector<int> label;
	for (int i = 0; i < testdata.rows; ++i) {
		cv::Mat data = forward(testdata.row(i));
		int tag = Argmax(data);
		label.push_back(tag);
	}

	float sum = 0;
	for (int i = 0; i < testlabel.rows; ++i) {
		if ((int)testlabel.at<float>(i, 0) == label.at(i)) {
			sum++;
		}
	}
	sum /= testlabel.rows;
	std::cout << "recoginition rate: " << sum<< std::endl;
	std::ofstream fout("res.txt");
	fout << sum;
	fout.close();
}

void NN::saveModel(std::string model_path)
{
	cv::FileStorage fs;
	fs.open(model_path + "w.xml", cv::FileStorage::WRITE);
	fs << "W" << NN_w;
	fs.release();

	fs.open(model_path + "b.xml", cv::FileStorage::WRITE);
	fs << "B" << NN_b;
	fs.release();
}

void NN::loadModel(std::string model_path)
{
	cv::FileStorage fs;
	fs.open(model_path + "w.xml", cv::FileStorage::READ);
	fs["W"] >> NN_w;
	fs.release();

	fs.open(model_path + "b.xml", cv::FileStorage::READ);
	fs["B"] >> NN_b;
	fs.release();
}

void NN::Mat_Shuffle(cv::Mat & data, cv::Mat & label)
{
	//矩阵打乱
	assert(data.rows == label.rows);
	std::default_random_engine e(time(0));
	std::uniform_int_distribution<unsigned> u(0, data.rows - 1);

	for (int i = data.rows - 1; i >= 0; --i) {
		int index = 0;
		do {
			index = u(e);
		} while (index == i);

		for (int j = 0; j < data.cols; ++j) {
			std::swap(data.at<float>(i, j), data.at<float>(index, j));
		}
		for (int j = 0; j < label.cols; ++j) {
			std::swap(label.at<float>(i, j), label.at<float>(index, j));
		}
	}
}

cv::Mat NN::RandMat(int rows, int cols)
{
	//生成一个随机矩阵，随机数范围为（0~1）;
	std::default_random_engine e(time(0));
	std::uniform_real_distribution<float> u(0, 0.1);

	cv::Mat temp = cv::Mat::zeros(rows, cols, CV_32FC1);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			temp.at<float>(i, j) = u(e);
		}
	}
	return temp;
}

cv::Mat NN::One_hot(cv::Mat mat, int K)
{
	//向量平铺矩阵;
	cv::Mat one_hot = cv::Mat::zeros(mat.rows, K, CV_32FC1);
	for (int i = 0; i < mat.rows; ++i) {
		int index = mat.at<float>(i, 0);
		one_hot.at<float>(i, index) = 1;
	}
	return one_hot;
}

inline int NN::Argmax(cv::Mat mat)
{
	assert(mat.rows == 1);
	float max = mat.at<float>(0, 0);
	int index = 0;

	for (int i = 0; i < mat.cols; ++i) {
		if (mat.at<float>(0, i) > max) {
			index = i;
			max = mat.at<float>(0, i);
		}
	}

	return index;
}

void NN::Mini_SGD(int mini_batch_begin)
{
	std::vector<cv::Mat> sum_der_W;
	std::vector<cv::Mat> sum_der_B;
	for (int i = 0; i < NN_size.size(); ++i) {
		cv::Mat temp_w = cv::Mat::zeros(NN_w.at(i).rows, NN_w.at(i).cols, CV_32FC1);
		sum_der_W.push_back(temp_w);
		cv::Mat temp_b = cv::Mat::zeros(NN_b.at(i).rows, NN_b.at(i).cols, CV_32FC1);
		sum_der_B.push_back(temp_b);
	}

	for (int i = mini_batch_begin; i < mini_batch_begin + para.batch_size; ++i) {
		backprop(i);

		assert(sum_der_W.size() == NN_der_w.size());
		int length = NN_der_w.size();
		for (int j = 0; j < NN_size.size(); ++j) {
			sum_der_W.at(j) += NN_der_w.at(length-1-j);
			sum_der_B.at(j) += NN_der_b.at(length - 1 - j);
		}
	}

	for (int i = 0; i < NN_size.size(); ++i) {
		sum_der_W.at(i) /= para.batch_size;
		sum_der_B.at(i) /= para.batch_size;
		NN_w.at(i) += para.alpha*sum_der_W.at(i);
		NN_b.at(i) += para.alpha*sum_der_B.at(i);
	}

}

cv::Mat NN::forward(cv::Mat data)
{
	for (int i = 0; i < NN_w.size()-1; ++i) {
		assert(data.cols == NN_w.at(i).rows);
		assert(NN_w.at(i).cols == NN_b.at(i).cols);
		data = data * NN_w.at(i) + NN_b.at(i);
		data = sigmoid(data);
	}
	assert(data.cols == NN_w.at(NN_w.size() - 1).rows);
	assert(NN_w.at(NN_w.size() - 1).cols == NN_b.at(NN_b.size() - 1).cols);
	assert(NN_w.at(NN_w.size() - 1).cols == trainlabel.cols);
	data = data * NN_w.at(NN_w.size() - 1)+NN_b.at(NN_b.size()-1);
	data = mean_exp(data);
	assert(data.rows == 1 && data.cols&&data.cols==para.classify_num);
	return data;
}

void NN::backprop(int index)
{
	NN_der_w.clear();
	NN_der_b.clear();

	//the process of forward;
	std::vector<cv::Mat> layer_a;
	std::vector<cv::Mat> layer_z;
	cv::Mat data = traindata.row(index);
	layer_a.push_back(data);
	for (int i = 0; i < NN_w.size() - 1; ++i) {
		data = data * NN_w.at(i) + NN_b.at(i);
		layer_z.push_back(data);
		data = sigmoid(data);
		layer_a.push_back(data);
	}
	data = data * NN_w.at(NN_w.size() - 1) + NN_b.at(NN_b.size() - 1);
	layer_z.push_back(data);
	data = mean_exp(data);
	layer_a.push_back(data);
	assert((layer_a.size() - 1) == layer_z.size());
	assert((layer_a.size() - 1) == NN_size.size());
	assert((layer_a.size() - 1) == NN_w.size());

	//update with the bp;
	int len = layer_z.size();

	assert(layer_a.at(len).rows == 1 && layer_a.at(len).cols == trainlabel.cols);
	cv::Mat sita_L = trainlabel.row(index) - layer_a.at(len);
	cv::Mat der_w = layer_a.at(len - 1).t()*sita_L;
	NN_der_w.push_back(der_w);
	NN_der_b.push_back(sita_L);

	cv::Mat sita_old = sita_L;
	for (int i = len - 2; i >=0; --i) {
		cv::Mat der_z = der_sigmoid(layer_z.at(i));
		cv::Mat sita = sita_old * NN_w.at(i+1).t();
		assert(sita.rows == der_z.rows&&sita.cols == der_z.cols);
		for (int i = 0; i < sita.rows; ++i) {
			for (int j = 0; j < sita.cols; ++j) {
				sita.at<float>(i, j) *= der_z.at<float>(i, j);
			}
		}
		assert(sita.rows == 1 && sita.cols == NN_w.at(i).cols);
		cv::Mat temp_der_w = layer_a.at(i).t()*sita;
		NN_der_w.push_back(temp_der_w);
		NN_der_b.push_back(sita);
		sita_old = sita;
	}

}

double NN::Loss()
{
	double loss = 0;
	for (int i = 0; i < traindata.rows; ++i) {
		cv::Mat data = forward(traindata.row(i));

		for (int k = 0; k < data.cols; ++k) {
			loss += (-(trainlabel.at<float>(i, k)*log(data.at<float>(0, k))));
		}
	}
	loss /= traindata.rows;
	return loss;
}

inline float NN::sigmoid(float z)
{
	return 1.0/(1.0+exp(-z));
}

cv::Mat NN::sigmoid(cv::Mat z)
{
	for (int i = 0; i < z.rows; ++i) {
		for (int j = 0; j < z.cols; ++j) {
			z.at<float>(i, j) = sigmoid(z.at<float>(i, j));
		}
	}

	return z;
}

cv::Mat NN::mean_exp(cv::Mat z)
{
	double sum=0;
	for (int i = 0; i < z.rows; ++i) {
		for (int j = 0; j < z.cols; ++j) {
			z.at<float>(i, j) = exp(z.at<float>(i, j));
			sum += z.at<float>(i, j);
		}
	}
	for (int i = 0; i < z.rows; ++i) {
		for (int j = 0; j < z.cols; ++j) {
			z.at<float>(i, j) = z.at<float>(i, j) / sum;
		}
	}
	return z;
}

inline float NN::der_sigmoid(float z)
{

	return sigmoid(z)*(1 - sigmoid(z));
}

cv::Mat NN::der_sigmoid(cv::Mat z)
{
	for (int i = 0; i < z.rows; ++i) {
		for (int j = 0; j < z.cols; ++j) {
			z.at<float>(i, j) = der_sigmoid(z.at<float>(i, j));
		}
	}
	return z;
}
