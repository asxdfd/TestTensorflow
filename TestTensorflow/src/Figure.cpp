#include "Figure.h"

Figure::Figure() { null = true; }

Figure::Figure(std::vector<cv::Point> &landmarks,
               std::vector<float> &headPose) {
  setLandmarks(landmarks);
  setHeadPose(headPose);
  null = false;
}

Figure::~Figure() {}

void Figure::setLandmarks(std::vector<cv::Point> &l) {
  landmarks.clear();
  for (cv::Point point : l) {
    landmarks.push_back(point);
  }
}

std::vector<cv::Point> Figure::getLandmarks() { return landmarks; }

void Figure::setHeadPose(std::vector<float> &h) {
  headPose.clear();
  for (float headpose : h) {
    headPose.push_back(headpose);
  }
}

std::vector<float> Figure::getHeadPose() { return headPose; }

bool Figure::is_null() { return null; }
