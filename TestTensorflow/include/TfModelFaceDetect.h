#pragma once

#include <fstream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "Figure.h"
#include "opencv2/core/core.hpp"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "TfModelLandmark.h"

class TfModelFaceDetect {
 public:
  TfModelFaceDetect();
  ~TfModelFaceDetect();
  int init();
  std::vector<Figure> predict(cv::Mat &);

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
  int width;
  int height;
  int channel;
  cv::Size size;
  int length;
  long shape[4];
  cv::Mat previous_image;
  std::vector<float[]> trick_boxes;
  std::vector<std::vector<cv::Point>> preLandmark;
  std::vector<float[]> tmp_box;
  TfModelLandmark tfModelLandmark;
  static tensorflow::Tensor Mat2Tensor(cv::Mat &, float);
  std::vector<float[]> simple_predict(cv::Mat &);
};