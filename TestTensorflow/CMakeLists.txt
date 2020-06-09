# CMakeList.txt : CMake project for TestTensorflow, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

include_directories("./include")
include_directories("D:/libtensorflow-cpu-windows-x86_64-1.15.0/include")
include_directories("E:/opencv-4.3.0-contrib/build_new/install/include")
# include_directories("E:/opencv/build/include")
link_directories("D:/libtensorflow-cpu-windows-x86_64-1.15.0/lib")
link_directories("E:/opencv-4.3.0-contrib/build_new/install/x64/vc16/lib")
# link_directories("E:/opencv/build/x64/vc15/lib")

# Add source to this project's executable.
add_executable (TestTensorflow "src/TestTensorflow.cpp" "src/TFUtils.cpp")

# TODO: Add tests and install targets if needed.
target_link_libraries(TestTensorflow
	tensorflow
	opencv_img_hash430
	opencv_world430
	# opencv_world430d
)