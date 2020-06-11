#pragma once

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

class Mat2Tensor {
 public:
  static tensorflow::Tensor mat2Tensor(cv::Mat& img, float normal = 1 / 255.0) {
    tensorflow::Tensor image_input = tensorflow::Tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape(
            {1, img.size().height, img.size().width, img.channels()}));

    float* tensor_data_ptr = image_input.flat<float>().data();
    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()),
                     tensor_data_ptr);
    img.convertTo(fake_mat, CV_32FC(img.channels()));

    // fake_mat *= normal;

    return image_input;
  }
};
