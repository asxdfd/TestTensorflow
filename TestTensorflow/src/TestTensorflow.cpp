// TestTensorflow.cpp : Defines the entry point for the application.
//

#include "TestTensorflow.h"

int main() {
  void testOpenCV();
  void testTensorflow();
  void testTensorflowCC();

  // testOpenCV();
  // testTensorflow();
  testTensorflowCC();
  return 0;
}

void testOpenCV() {
  cv::VideoCapture v(0);
  cv::CascadeClassifier face_cascade(
      "D:/EB/theia/faceAnalysis/model/haarcascade_frontalface_alt2.xml");
  cv::Ptr<cv::face::Facemark> facemark;
  facemark = cv::face::FacemarkLBF::create();
  facemark->loadModel("D:/EB/theia/faceAnalysis/model/lbfmodel.yaml");

  while (1) {
    // Grab a frame
    cv::Mat img;
    v >> img;

    if (img.empty()) break;

    cv::Mat gray;
    // while (stream.isOpened()) {
    // 存储人脸矩形框的容器
    std::vector<cv::Rect> faces;
    // 将视频帧转换至灰度图, 因为Face Detector的输入是灰度图
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // 人脸检测
    face_cascade.detectMultiScale(gray, faces);
    if (!faces.empty()) {
      // 人脸关键点的容器
      std::vector<std::vector<cv::Point2f>> landmarks;

      // 运行人脸关键点检测器（landmark detector）
      bool success = facemark->fit(img, faces, landmarks);

      if (success) {
        for (int i = 0; i < landmarks.size(); i++) {
          // 自定义绘制人脸特征点函数, 可绘制人脸特征点形状/轮廓
          // drawLandmarks(frame, landmarks[i]);
          // OpenCV自带绘制人脸关键点函数: drawFacemarks
          cv::face::drawFacemarks(img, landmarks[i], cv::Scalar(0, 0, 255));
        }
      }
    } else {
    }

    // Display the resulting frame
    cv::imshow("Frame", img);

    // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(25);
    if (c == 27) break;
  }
}

static void BoolDeallocator(void* data, size_t, void* arg) {
  delete[] static_cast<bool*>(data);
}

void testTensorflow() {
  const std::string PB_FILE_PATH =
      "D:/EB/theia/faceAnalysis/model/faceboxes/detector.pb";
  const std::string INPUT_TENSOR_NAME = "tower_0/images";
  const std::string OUTPUT_TENSOR_NAME1 = "tower_0/boxes";
  const std::string OUTPUT_TENSOR_NAME2 = "tower_0/scores";
  const std::string OUTPUT_TENSOR_NAME3 = "tower_0/num_detections";

  cv::VideoCapture v(0);
  TFUtils TFU;
  TFUtils::STATUS status = TFU.LoadModel(PB_FILE_PATH);

  if (status != TFUtils::SUCCESS) {
    std::cerr << "Can't load graph" << std::endl;
    return;
  }
  while (1) {
    // Grab a frame
    cv::Mat img;
    v >> img;

    if (img.empty()) break;

    const std::vector<std::int64_t> input_dims = {
        1, img.size().height, img.size().width, img.channels()};

    TF_Tensor* input_image = TFUtils::Mat2Tensor(img, 1 / 255.0);

    // Input Tensor/Ops Create
    std::vector<std::int64_t> dim;
    dim.push_back(1);
    std::vector<bool> d;
    d.push_back(false);
    const std::vector<TF_Tensor*> input_tensors = {
        input_image, TF_NewTensor(TF_BOOL, nullptr, 0, {false}, sizeof(bool),
                                  &BoolDeallocator, nullptr)};

    const std::vector<TF_Output> input_ops = {
        TFU.GetOperationByName(INPUT_TENSOR_NAME, 0),
        TFU.GetOperationByName("training_flag", 0)};

    // Output Tensor/Ops Create
    const std::vector<TF_Output> output_ops = {
        TFU.GetOperationByName(OUTPUT_TENSOR_NAME1, 0),
        TFU.GetOperationByName(OUTPUT_TENSOR_NAME2, 0),
        TFU.GetOperationByName(OUTPUT_TENSOR_NAME3, 0)};

    std::vector<TF_Tensor*> output_tensors = {nullptr, nullptr, nullptr};

    status =
        TFU.RunSession(input_ops, input_tensors, output_ops, output_tensors);
    std::cout << "run session" << std::endl;
    if (status == TFUtils::SUCCESS) {
      const std::vector<std::vector<std::vector<float>>> data =
          TFUtils::GetTensorsData<std::vector<float>>(output_tensors);
      const std::vector<std::vector<float>> result = data[0];
      std::cout << "data.size = " << data.size() << std::endl;
      int i = 0;
      for (auto a : data) {
        std::cout << "data[" << i << "].size = " << a.size() << std::endl;
        int j = 0;
        for (auto b : a) {
          std::cout << "data[" << i << "][" << j << "].size = " << b.size()
                    << std::endl;
          int k = 0;
          for (auto c : b) {
            std::cout << "data[" << i << "][" << j << "][" << k << "] = " << c
                      << std::endl;
            k++;
          }
          j++;
        }
        i++;
      }
    } else {
      std::cout << "Error run session";
      return;
    }

    TFUtils::DeleteTensors(input_tensors);
    TFUtils::DeleteTensors(output_tensors);

    // Display the resulting frame
    cv::imshow("Frame", img);

    // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(25);
    if (c == 27) break;
  }
}

void testTensorflowCC() {
  TfModelFaceDetect face;

  int ret = face.init();
  if (ret != 0) {
    std::cout << "face init error" << std::endl;
    return;
  }

  cv::VideoCapture v(0);

  while (1) {
    // Grab a frame
    cv::Mat img;
    v >> img;

    if (img.empty()) break;

    std::vector<Figure> figures = face.predict(img);
    for (Figure figure : figures) {
      std::vector<cv::Point> landmarks = figure.getLandmarks();
      // 自定义绘制人脸特征点函数, 可绘制人脸特征点形状/轮廓
      // drawLandmarks(frame, landmarks[i]);
      // OpenCV自带绘制人脸关键点函数: drawFacemarks
      cv::face::drawFacemarks(img, landmarks, cv::Scalar(0, 0, 255));
    }
    // Display the resulting frame
    cv::imshow("Frame", img);

    // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(25);
    if (c == 27) break;
  }
}
