#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <json/json.h>
#include "camera.h"
#include "math_util.h"


const int JOINT_SIZE = 21;
const Eigen::Matrix2Xi BONE = [] {
	Eigen::Matrix2Xi bone(2, 20);
	bone << 0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
		1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20;
	return bone;
}();


std::vector<std::vector<Eigen::Matrix4Xf>> LoadSkels(const std::string& filename)
{
	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "file not exist: " << filename << std::endl;
		std::abort();
	}

	int frameSize, personSize;
	fs >> frameSize;
	std::vector<std::vector<Eigen::Matrix4Xf>> skels(frameSize);
	for (int frameIdx = 0; frameIdx < frameSize; frameIdx++) {
		fs >> personSize;
		skels[frameIdx].resize(personSize, Eigen::Matrix4Xf::Zero(4, JOINT_SIZE));
		for (int pIdx = 0; pIdx < personSize; pIdx++)
			for (int i = 0; i < 4; i++)
				for (int jIdx = 0; jIdx < JOINT_SIZE; jIdx++)
					fs >> skels[frameIdx][pIdx](i, jIdx);
	}
	fs.close();
	return skels;
}


Eigen::Matrix2Xi LoadSyncPoints(const std::string& filename) {
	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "file not exist: " << filename << std::endl;
		std::abort();
	}

	int cnt;
	fs >> cnt;
	Eigen::Matrix2Xi syncPoints(2, cnt);
	for (int i = 0; i < cnt; i++)
		fs >> syncPoints(0, i) >> syncPoints(1, i);
	fs.close();
	return syncPoints;
}


const cv::Scalar& GetColor(const int& idx)
{
	static const std::vector<cv::Scalar> colorVec = {
		{ 230, 216, 173 },					// light blue
		{ 210, 250, 250 },					// light_golden
		{ 193, 182, 255 },					// light_pink
		{ 128, 128, 240 },					// light_coral
		{ 144, 238, 144 },					// light_green
		{ 255, 191, 0 },					// deep_sky_blue
		{ 113, 179, 60 },					// medium_sea_green
		{ 122, 160, 255 },					// light_salmon	
		{ 221, 160, 221 },					// plum
		{ 255, 255, 0 },					// cyan
		{ 212, 255, 127 },					// aqua_marine
		{ 250, 206, 135 },					// light_sky_blue
		{92, 92, 205},						// indian red
		{ 230, 245, 253 },					// old_lace
		{ 180, 105, 255 },					// hot_pink
		{71, 99, 255},						// tomato
		{250, 230, 230}						// lavender
	};
	return colorVec[(idx + colorVec.size()) % colorVec.size()];
}


void DrawSkel(const Eigen::Matrix4Xf& skel3d, const Eigen::Matrix<float, 3, 4>& proj, const cv::Scalar& color, cv::Mat img)
{
	cv::Size size(img.cols, img.rows);
	Eigen::Matrix3Xf skel2d(3, JOINT_SIZE);
	skel2d.topRows(2) = (proj * skel3d.topRows(3).colwise().homogeneous()).colwise().hnormalized();
	skel2d.row(2) = skel3d.row(3);

	for (int jIdx = 0; jIdx < JOINT_SIZE; jIdx++) {
		if (skel2d(2, jIdx) < FLT_EPSILON)
			continue;

		cv::Point jPos = MathUtil::Vec2Point(skel2d.col(jIdx).head(2), size);
		cv::circle(img, jPos, 5, color, 1);
		cv::putText(img, std::to_string(jIdx), jPos, cv::FONT_HERSHEY_PLAIN, 1.f, color);
	}


	for (int boneIdx = 0; boneIdx < BONE.cols(); boneIdx++) {
		const int jaIdx = BONE(0, boneIdx);
		const int jbIdx = BONE(1, boneIdx);
		if (skel2d(2, jaIdx) < FLT_EPSILON || skel2d(2, jbIdx) < FLT_EPSILON)
			continue;

		cv::line(img, MathUtil::Vec2Point(skel2d.col(jaIdx).head(2), size),
			MathUtil::Vec2Point(skel2d.col(jbIdx).head(2), size), color, 3);
	}
}


int main()
{
	const std::string& dataset = "seq2";
	std::map<std::string, Camera> cameraMap = ParseCameras("../dataset/calibration.json");
	std::vector<std::vector<Eigen::Matrix4Xf>> skels3d = LoadSkels("../dataset/" + dataset + "/gt.txt");
	Eigen::Matrix2Xi syncPoints = LoadSyncPoints("../dataset/" + dataset + "/sync_points.txt");

	std::vector<Camera> cameras(cameraMap.size());
	std::vector<cv::VideoCapture> videos(cameraMap.size());
	std::vector<cv::Mat> rawImgs(cameraMap.size());
	const int startFrame = 0;

	for (int i = 0; i < cameraMap.size(); i++) {
		auto iter = std::next(cameraMap.begin(), i);
		cameras[i] = iter->second;
		videos[i] = cv::VideoCapture("../dataset/" + dataset + "/" + iter->first + ".mp4");
		videos[i].set(cv::CAP_PROP_POS_FRAMES, startFrame);
		rawImgs[i] = cv::Mat();
	}

	int syncIdx = 0;
	for (int frameIdx = startFrame; ; frameIdx++) {
		for (int view = 0; view < cameras.size(); view++) {
			videos[view] >> rawImgs[view];
			if (rawImgs[view].empty())
				return 0;
		}

		while (frameIdx >= syncPoints(0, syncIdx))
			syncIdx++;

		const int gtIdx = syncPoints(1, syncIdx - 1) + int(std::round(float(frameIdx - syncPoints(0, syncIdx - 1)) *
			(float(syncPoints(1, syncIdx) - syncPoints(1, syncIdx - 1)) / float(syncPoints(0, syncIdx) - syncPoints(0, syncIdx - 1)))));
			
		const int layoutCols = 3;
		const int layoutRows = int((cameras.size() + layoutCols - 1) / layoutCols);
		const cv::Size sliceSize(512, 512);
		cv::Mat mergedImg(layoutRows*sliceSize.height, layoutCols*sliceSize.width, CV_8UC3);
		std::vector<cv::Rect> rois;
		for (int view = 0; view < cameras.size(); view++) {
			cv::Rect roi(cv::Point2i(view%layoutCols * sliceSize.width, view / layoutCols * sliceSize.height), sliceSize);
			cv::resize(rawImgs[view], mergedImg(roi), sliceSize);
			rois.emplace_back(roi);
		}
		
		for (int view = 0; view < cameras.size(); view++)
			for (int pIdx = 0; pIdx < skels3d[gtIdx].size(); pIdx++)
				DrawSkel(skels3d[gtIdx][pIdx], cameras[view].proj, GetColor(pIdx), mergedImg(rois[view]));


		cv::imwrite("../output/" + std::to_string(frameIdx) + ".jpg", mergedImg);
		std::cout << std::to_string(frameIdx) << std::endl;
	}
	return 0;
}

