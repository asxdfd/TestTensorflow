#pragma once

#include <fstream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "Figure.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "Mat2Tensor.h"
#include "TfModelLandmark.h"

class ReShapeImg {
 public:
  cv::Mat rs_mat;
  float scale_x;
  float scale_y;
  ReShapeImg(cv::Mat &);
  ~ReShapeImg();
};

class TfModelFaceDetect {
 public:
  static int width;
  static int height;
  static int channel;
  static cv::Size size;
  TfModelFaceDetect();
  ~TfModelFaceDetect();
  int init();
  std::vector<Figure> predict(cv::Mat &);
  std::vector<std::vector<float>> simple_predict(ReShapeImg &);

 private:
  tensorflow::GraphDef graph;
  tensorflow::Session *session;
  std::string PB_FILE_PATH;
  std::string INPUT_TENSOR_NAME;
  std::string OUTPUT_TENSOR_NAME1;
  std::string OUTPUT_TENSOR_NAME2;
  std::string OUTPUT_TENSOR_NAME3;
  tensorflow::Tensor flag;
  float thres;
  int batch;
  int length;
  long shape[4];
  cv::Mat previous_image;
  std::vector<std::vector<float>> trick_boxes;
  std::vector<std::vector<cv::Point>> preLandmark;
  std::vector<std::vector<float>> tmp_box;
  TfModelLandmark tfModelLandmark;
};
