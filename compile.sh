rm -rf build/
mkdir -p build/
cd build/
cmake ..
make -j4
cp webcam-detector ../