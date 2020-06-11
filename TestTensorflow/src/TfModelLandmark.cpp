#include "TfModelLandmark.h"

int TfModelLandmark::batch = 1;
int TfModelLandmark::width = 160;
int TfModelLandmark::height = 160;
int TfModelLandmark::channel = 3;
int TfModelLandmark::length = width * height * channel;
long TfModelLandmark::shape[] = {batch, height, width, channel};

TfModelLandmark::TfModelLandmark() {
  min_face = 20;
  point_num = 68 * 2;
  base_extend_range[0] = 0.2f;
  base_extend_range[1] = 0.3f;
  size = cv::Size(160, 160);
  PB_FILE_PATH =
      "D:/EB/theia/faceAnalysis/model/keypoints/shufflenet/keypoints.pb";
  INPUT_TENSOR_NAME = "tower_0/images:0";
  OUTPUT_TENSOR_NAME = "tower_0/prediction:0";
  flag = tensorflow::Tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
  flag.scalar<bool>()() = false;
}

TfModelLandmark::~TfModelLandmark() {
  if (session != nullptr) session->Close();
}

int TfModelLandmark::init() {
  tensorflow::Status status =
      NewSession(tensorflow::SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "TfModelLandmark: Session created successfully" << std::endl;
  }

  // Load the protobuf graph
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), PB_FILE_PATH,
                                       &graph);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "TfModelLandmark: Load graph protobuf successfully"
              << std::endl;
  }

  // Add the graph to the session
  status = session->Create(graph);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "TfModelLandmark: Add graph to session successfully"
              << std::endl;
    return 0;
  }
}

std::vector<Figure> TfModelLandmark::predict(
    cv::Mat& mat, std::vector<std::vector<float>>& boxes) {
  std::vector<Figure> figures;

  for (std::vector<float> bbox : boxes) {
    Figure a = onShotRun(mat, bbox);
    if (!a.is_null()) {
      figures.push_back(a);
    }
  }

  return figures;
}

Figure TfModelLandmark::onShotRun(cv::Mat& mat, std::vector<float> bbox) {
  float bbox_width = bbox[2] - bbox[0];
  float bbox_height = bbox[3] - bbox[1];

  if (bbox_width <= min_face || bbox_height <= min_face) {
    return Figure();
  } else {
    int add = bbox_width > bbox_height ? (int)bbox_width : (int)bbox_height;
    cv::Mat bimg;
    cv::copyMakeBorder(mat, bimg, add, add, add, add, cv::BORDER_CONSTANT,
                       cv::Scalar(103, 116, 123, 0));

    for (int i = 0; i < bbox.size(); i++) {
      bbox[i] += add;
    }
    int half_edge = ((int)((1 + 2 * base_extend_range[0]) * bbox_width)) >> 1;
    int center[] = {(int)(bbox[0] + bbox[2]) / 2, (int)(bbox[1] + bbox[3]) / 2};

    bbox[0] = center[0] - half_edge;
    bbox[1] = center[1] - half_edge;
    bbox[2] = center[0] + half_edge;
    bbox[3] = center[1] + half_edge;

    /**
     * (0,1)  (2-0,3-1)
     * */
    // Ωÿ≤ø∑÷Õº
    cv::Mat crop_image(
        bimg, cv::Rect((int)bbox[0], (int)bbox[1], (int)(bbox[2] - bbox[0]),
                       (int)(bbox[3] - bbox[1])));

    int h = crop_image.rows;
    int w = crop_image.cols;
    cv::Mat r_crop_img;
    cv::resize(crop_image, r_crop_img, size);

    MarkFeature mf = simple_predict(r_crop_img);

    //            System.out.println(Arrays.toString(mf.landmark_x.toFloatVector()));
    for (int i = 0; i < mf.landmark_x.size(); i++) {
      mf.landmark_x[i] =
          mf.landmark_x[i] * w / size.width * size.height + bbox[0] - add;
    }
    for (int i = 0; i < mf.landmark_y.size(); i++) {
      mf.landmark_y[i] =
          mf.landmark_y[i] * h / size.height * size.width + bbox[1] - add;
    }

    std::vector<cv::Point> lm = mf.landmark();
    std::vector<float> hp = mf.headpose();
    return Figure(lm, hp);
  }
}

MarkFeature TfModelLandmark::simple_predict(cv::Mat& mat) {
  tensorflow::Tensor data = Mat2Tensor::mat2Tensor(mat);
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {INPUT_TENSOR_NAME, data}, {"training_flag:0", flag}};
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  tensorflow::Status status =
      session->Run(inputs, {OUTPUT_TENSOR_NAME}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;

    return MarkFeature();
  } else {
    // std::cout << "TfModelFaceDetect: Run session successfully" << std::endl;
  }

  const tensorflow::TTypes<float, 2>::Tensor& embeddings =
      outputs[0].flat_inner_dims<float, 2>();
  std::vector<std::vector<float>> landmark;
  std::vector<float> headpose;
  for (int i = 0; i < 68 * 2; i++) {
    std::vector<float> l;
    l.push_back(embeddings(0, i++));
    l.push_back(embeddings(0, i));
    landmark.push_back(l);
  }
  for (int i = 68 * 2; i < 68 * 2 + 3; i++) {
    headpose.push_back(embeddings(0, i));
  }

  return MarkFeature(landmark, headpose);
}

MarkFeature::MarkFeature() {}

MarkFeature::MarkFeature(std::vector<std::vector<float>>& l, std::vector<float>& h) {
  for (std::vector<float> landmark : l) {
    landmark_x.push_back(landmark[0]);
    landmark_y.push_back(landmark[1]);
  }
  for (int i = 0; i < 3; i++) {
    headPose.push_back(h[i]);
  }
}

MarkFeature::~MarkFeature() {}

std::vector<cv::Point> MarkFeature::landmark() {
  std::vector<cv::Point> p;
  for (int i = 0; i < 68; i++) {
    cv::Point point((int)landmark_x[i], (int)landmark_y[i]);
    p.push_back(point);
  }

  return p;
}

std::vector<float> MarkFeature::headpose() { return headPose; }
