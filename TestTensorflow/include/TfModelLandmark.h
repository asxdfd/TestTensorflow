#pragma once

#include <vector>
#include "Figure.h"
#include "opencv2/core/core.hpp"

class TfModelLandmark {
 public:
  TfModelLandmark();
  ~TfModelLandmark();
  std::vector<Figure> predict(cv::Mat &, std::vector<float[]>);
};