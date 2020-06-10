#include "TfModelFaceDetect.h"

TfModelFaceDetect::TfModelFaceDetect() {
  PB_FILE_PATH = "./model/faceboxes/detector.pb";
  INPUT_TENSOR_NAME = "tower_0/images:0";
  OUTPUT_TENSOR_NAME1 = "tower_0/boxes:0";
  OUTPUT_TENSOR_NAME2 = "tower_0/scores:0";
  OUTPUT_TENSOR_NAME3 = "tower_0/num_detections:0";
  flag = tensorflow::Tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
  flag.scalar<bool>()() = false;
  thres = 0.5f;
  batch = 1;
  width = 512;
  height = 512;
  channel = 3;
  size = cv::Size(width, height);
  length = width * height * channel;
  shape[0] = batch;
  shape[1] = height;
  shape[2] = width;
  shape[3] = channel;
}

TfModelFaceDetect::~TfModelFaceDetect() {}

int TfModelFaceDetect::init() {
  tensorflow::Status status =
      NewSession(tensorflow::SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Session created successfully" << std::endl;
  }

  // Load the protobuf graph
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), PB_FILE_PATH,
                                       &graph);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Load graph protobuf successfully" << std::endl;
  }

  // Add the graph to the session
  status = session->Create(graph);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Add graph to session successfully" << std::endl;
    return 0;
  }
}

std::vector<float[]> TfModelFaceDetect::simple_predict(cv::Mat &mat) {
  tensorflow::Tensor data = Mat2Tensor(mat);
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {INPUT_TENSOR_NAME, data}, {"training_flag:0", flag}};
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  tensorflow::Status status = session->Run(
      inputs, {OUTPUT_TENSOR_NAME1, OUTPUT_TENSOR_NAME2, OUTPUT_TENSOR_NAME3},
      {}, &outputs);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    std::vector<float[]> res;
    return res;
  } else {
    std::cout << "Run session successfully" << std::endl;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  // Print the results
  std::cout << outputs[0].DebugString()
            << std::endl;  // Tensor<type: float shape: [] values: 30>

  //    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
  //    Eigen::Aligned>& prediction = outputs[0].flat<float>();
  const tensorflow::TTypes<float, 1>::Tensor &prediction =
      outputs[0].flat_inner_dims<float, 1>();

  session->Close();

  std::vector<float[]> res;
  return res;
}

std::vector<Figure> TfModelFaceDetect::predict(cv::Mat &mat) {
  std::vector<float[]> boxes = simple_predict(mat);
  std::vector<Figure> figures = tfModelLandmark.predict(mat, boxes);

  std::vector<std::vector<cv::Point>> landmarks;
  for (Figure figure : figures) {
    landmarks.push_back(figure.getLandmarks());
  }

  return figures;
}

tensorflow::Tensor TfModelFaceDetect::Mat2Tensor(cv::Mat &img,
                                                 float normal = 1 / 255.0) {
  tensorflow::Tensor image_input = tensorflow::Tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape(
          {1, img.size().height, img.size().width, img.channels()}));

  float *tensor_data_ptr = image_input.flat<float>().data();
  cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()),
                   tensor_data_ptr);
  img.convertTo(fake_mat, CV_32FC(img.channels()));

  fake_mat *= normal;

  return image_input;
}
