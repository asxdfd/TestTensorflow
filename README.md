# TestTensorflow

## TODOS
- [x] 使用OpenCV完成关键点识别
- [ ] 使用Tensorflow C API完成关键点识别
- [x] 使用Tensorflow C++ API完成关键点识别
- [ ] 将代码迁移到UE4中

## TestOpenCV
- 使用OpenCV完成关键点识别
- 使用cv::CascadeClassifier和FacemarkLBF
- 使用方法：在main函数中调用
  ```
  testOpenCV();
  ```

## ~~TestTensorflow（无法使用）~~
- 使用Tensorflow C API完成关键点识别
- 下载地址：[libtensorflow](https://tensorflow.google.cn/install/lang_c)
- 工具类TFUtils：封装了Tensorflow C API
- 使用方法：在main函数中调用
  ```
  testTensorflow();
  ```
- TFUtils使用方法：[参考链接](http://www.liuxiao.org/2018/12/tensorflow-c-api-%e4%bb%8e%e8%ae%ad%e7%bb%83%e5%88%b0%e9%83%a8%e7%bd%b2%ef%bc%9a%e4%bd%bf%e7%94%a8-c-api-%e8%bf%9b%e8%a1%8c%e9%a2%84%e6%b5%8b%e5%92%8c%e9%83%a8%e7%bd%b2/)
  - 创建类实例
    ```
    TFUtils TFU;
    ```
  - 加载模型
    ```
    TFUtils::STATUS status = TFU.LoadModel(PB_FILE_PATH);

    if (status != TFUtils::SUCCESS) {
      std::cerr << "Can't load graph" << std::endl;
      return;
    }
    ```
  - 搭建网络
    ```
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
    ```
    GetOperationByName中，name与pb模型中对应。
  - 运行网络
    ```
    status = TFU.RunSession(input_ops, input_tensors, output_ops, output_tensors);
    ```
  - 得到输出
    ```
    if (status == TFUtils::SUCCESS) {
      const std::vector<std::vector<T>> data = TFUtils::GetTensorsData<T(output_tensors);
    }
    ```
- 存在的bug：No symbol file loaded for tensorflow.dll

## TestTensorflowCC
- 使用Tensorflow C++ API完成关键点识别
- 使用方法：在main函数中调用
  ```
  testTensorflowCC();
  ```
- 已封装好的类：TfModelFaceDetect、TfModelLandmark
  - 初始化，初始化成功返回0
    ```
    TfModelFaceDetect face;

    int ret = face.init();
    if (ret != 0) {
      std::cout << "face init error" << std::endl;
      return;
    }
    ```
  - 预测并绘制结果
    ```
    std::vector<Figure> figures = face.predict(img);
    for (Figure figure : figures) {
      std::vector<cv::Point> landmarks = figure.getLandmarks();
      cv::face::drawFacemarks(img, landmarks, cv::Scalar(0, 0, 255));
    }
    ```
  - 其他：cv::Mat转换为tensorflow::Tensor
    ```
    #include "Mat2Tensor.h"
    
    // get a cv::Mat object
    Mat2Tensor::mat2Tensor(mat);
    ```
- [参考资料](http://www.liuxiao.org/2018/10/tensorflow-c-%e4%bb%8e%e8%ae%ad%e7%bb%83%e5%88%b0%e9%83%a8%e7%bd%b23%ef%bc%9a%e4%bd%bf%e7%94%a8-keras-%e8%ae%ad%e7%bb%83%e5%92%8c%e9%83%a8%e7%bd%b2-cnn/)
