#pragma once

#include <vector>

#include "Figure.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "Mat2Tensor.h"

class MarkFeature {
 public:
  std::vector<float> landmark_x;
  std::vector<float> landmark_y;
  std::vector<float> headPose;
  MarkFeature();
  MarkFeature(std::vector<std::vector<float>> &, std::vector<float> &);
  ~MarkFeature();
  std::vector<cv::Point> landmark();
  std::vector<float> headpose();
};

class TfModelLandmark {
 public:
  static int batch;
  static int width;
  static int height;
  static int channel;
  static int length;
  static long shape[4];
  TfModelLandmark();
  ~TfModelLandmark();
  int init();
  std::vector<Figure> predict(cv::Mat &, std::vector<std::vector<float>> &);

 private:
  tensorflow::GraphDef graph;
  tensorflow::Session *session;
  int min_face;
  int point_num;
  float base_extend_range[2];
  cv::Size size;
  std::string PB_FILE_PATH;
  std::string INPUT_TENSOR_NAME;
  std::string OUTPUT_TENSOR_NAME;
  tensorflow::Tensor flag;
  Figure onShotRun(cv::Mat &, std::vector<float>);
  MarkFeature simple_predict(cv::Mat &);
};
