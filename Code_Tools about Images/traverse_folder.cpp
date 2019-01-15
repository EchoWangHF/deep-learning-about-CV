#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<io.h>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

int main() {
	
	int i = 1;
	struct _finddata_t fileinfo;
	string path = "E:\\DataSet\\过车图片\\*.*";       //修改源文件的地址
	intptr_t hFile;
	if ((hFile = _findfirst(path.c_str(), &fileinfo)) == -1)
		return -1;
	else {
		do {
			string name = fileinfo.name;     //遍历获取文件名

			
			string imgpath = "E:\\DataSet\\过车图片\\" + name;
			Mat img = imread(imgpath, 1);
			if (img.empty()) {
				std::cout << "still...\n";
				continue;
			}
			else
			{
				cv::Rect rect(0, 0, img.cols, img.rows - 70);
				cv::Mat img_temp = img(rect).clone();
				cv::resize(img_temp, img_temp, cv::Size(img_temp.cols*0.5,img_temp.rows*0.5));

				string writepath = "E:\\DataSet\\Img_Pre\\" + to_string(i) + ".jpg";  //修改后文件的存放地址
				imwrite(writepath, img_temp);
				i++;
			}
			

		} while (_findnext(hFile, &fileinfo) == 0);
	}
	_findclose(hFile);
	return 0;
}
