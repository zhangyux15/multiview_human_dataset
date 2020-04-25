#pragma once
#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <json/json.h>


struct Camera
{
	// origin size
	cv::Matx33f cvK, cvNewK, cvR;
	cv::Vec3f cvT;
	cv::Matx<float, 5, 1> cvDistCoeff = cv::Matx<float, 5, 1>::zeros();
	cv::Size cvImgSize, cvNewImgSize = cv::Size(0, 0);
	cv::Rect cvValidPixROI = cv::Rect(0, 0, 0, 0);
	double cvRectifyAlpha = 0;
	cv::Mat cvRectifyMapX, cvRectifyMapY;

	// normalized
	Eigen::Matrix3f K, Ki, R, Rt, RtKi;
	Eigen::Vector3f T, pos;
	Eigen::Matrix<float, 3, 4> proj;

	Camera() {}
	Camera(const Json::Value& json) { Parse(json); }
	void Parse(const Json::Value& json);
	Json::Value Serialize() const;

	void Update();
	void LookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up);
	Eigen::Matrix3f CalcFundamental(const Camera& camera) const;
	Eigen::Vector3f CalcRay(const Eigen::Vector2f& uv) const;
};


std::map<std::string, Camera> ParseCameras(const std::string& jsonFile);
void SerializeCameras(const std::map<std::string, Camera>& cameras, const std::string& jsonFile);


struct Triangulator
{
	Eigen::Matrix3Xf points;
	Eigen::Matrix3Xf projs;
	bool convergent;
	float loss;
	Eigen::Vector3f pos;
	void Solve(const int& maxIterTime = 20, const float& updateTolerance = 1e-4f, const float& regularTerm = 1e-4f);
};