#include "TfModelFaceDetect.h"

int TfModelFaceDetect::width = 512;
int TfModelFaceDetect::height = 512;
int TfModelFaceDetect::channel = 3;
cv::Size TfModelFaceDetect::size(width, height);

TfModelFaceDetect::TfModelFaceDetect() {
  PB_FILE_PATH = "D:/EB/theia/faceAnalysis/model/faceboxes/detector.pb";
  INPUT_TENSOR_NAME = "tower_0/images:0";
  OUTPUT_TENSOR_NAME1 = "tower_0/boxes:0";
  OUTPUT_TENSOR_NAME2 = "tower_0/scores:0";
  OUTPUT_TENSOR_NAME3 = "tower_0/num_detections:0";
  flag = tensorflow::Tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
  flag.scalar<bool>()() = false;
  thres = 0.5f;
  batch = 1;
  length = width * height * channel;
  shape[0] = batch;
  shape[1] = height;
  shape[2] = width;
  shape[3] = channel;
}

TfModelFaceDetect::~TfModelFaceDetect() {
  if (session != nullptr) session->Close();
}

int TfModelFaceDetect::init() {
  tensorflow::Status status =
      NewSession(tensorflow::SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "TfModelFaceDetect: Session created successfully" << std::endl;
  }

  // Load the protobuf graph
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), PB_FILE_PATH,
                                       &graph);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "TfModelFaceDetect: Load graph protobuf successfully" << std::endl;
  }

  // Add the graph to the session
  status = session->Create(graph);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "TfModelFaceDetect: Add graph to session successfully" << std::endl;
  }

  return tfModelLandmark.init();
}

std::vector<std::vector<float>> TfModelFaceDetect::simple_predict(
    ReShapeImg &reShapeImg) {
  tensorflow::Tensor data = Mat2Tensor::mat2Tensor(reShapeImg.rs_mat);
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {INPUT_TENSOR_NAME, data}, {"training_flag:0", flag}};
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  tensorflow::Status status = session->Run(
      inputs, {OUTPUT_TENSOR_NAME1, OUTPUT_TENSOR_NAME2, OUTPUT_TENSOR_NAME3},
      {}, &outputs);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    std::vector<std::vector<float>> res;
    return res;
  } else {
    //std::cout << "TfModelFaceDetect: Run session successfully" << std::endl;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  // Print the results
  // std::cout << outputs[0].DebugString() << std::endl;
  // std::cout << outputs[1].DebugString() << std::endl;
  // std::cout << outputs[2].DebugString() << std::endl;

  //    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
  //    Eigen::Aligned>& prediction = outputs[0].flat<float>();
  const tensorflow::TTypes<float, 3>::Tensor &layer1 =
      outputs[0].flat_inner_dims<float, 3>();
  const tensorflow::TTypes<float, 2>::Tensor &layer2 =
      outputs[1].flat_inner_dims<float, 2>();
  const int layer3 = outputs[2].flat_inner_dims<int, 1>()(0);
  // session->Close();

  float scaler[] = {height / reShapeImg.scale_y, width / reShapeImg.scale_x,
                    height / reShapeImg.scale_y, width / reShapeImg.scale_x};
  std::vector<std::vector<float>> boxes;
  for (int i = 0; i < layer3; i++) {
    if (layer2(0, i) > thres) {
      std::vector<float> box;
      box.push_back(layer1(0, i, 1) * scaler[1]);
      box.push_back(layer1(0, i, 0) * scaler[0]);
      box.push_back(layer1(0, i, 3) * scaler[3]);
      box.push_back(layer1(0, i, 2) * scaler[2]);
      boxes.push_back(box);
    }
  }

  return boxes;
}

std::vector<Figure> TfModelFaceDetect::predict(cv::Mat &mat) {
  std::vector<std::vector<float>> boxes = simple_predict(ReShapeImg(mat));
  std::vector<Figure> figures = tfModelLandmark.predict(mat, boxes);

  std::vector<std::vector<cv::Point>> landmarks;
  for (Figure figure : figures) {
    landmarks.push_back(figure.getLandmarks());
  }

  return figures;
}

ReShapeImg::ReShapeImg(cv::Mat &src) {
  int h = src.rows;
  int w = src.cols;
  int long_side = h >= w ? h : w;
  scale_y = (float)TfModelFaceDetect::height / long_side;
  scale_x = scale_y;
  rs_mat = cv::Mat(TfModelFaceDetect::height, TfModelFaceDetect::width, CV_8UC3,
                   cv::Scalar(103, 116, 123, 0));
  std::vector<int> size = {(int)(w * scale_x), (int)(h * scale_y)};
  cv::Mat rm;
  cv::resize(src, rm, cv::Size((int)(w * scale_x), (int)(h * scale_y)));
  rm.copyTo(rs_mat(cv::Rect(0, 0, rm.cols, rm.rows)));
  rm.release();
}

ReShapeImg::~ReShapeImg() { rs_mat.release(); }
