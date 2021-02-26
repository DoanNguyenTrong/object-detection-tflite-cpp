CXXFLAGS ?= -IC:/dev/godev/src/github.com/tensorflow/tensorflow -Ic:/msys64/mingw64/include/opencv4 -Ic:/msys64/mingw64/include/freetype2
LDFLAGS ?= -LC:/dev/godev/src/github.com/tensorflow/tensorflow/tensorflow/lite/tools/make/gen/windows_x86_64/lib

.PHONY: all clean

all: webcam-detector

webcam-detector: main.o
	gcc -O3 -o webcam-detector main.o $(LDFLAGS) -ltensorflow-lite -lstdc++ -lpthread -ldl -lm -lopencv_videoio -lopencv_core -lopencv_highgui -lopencv_imgproc -lfreetype

main.o : main.cxx
	g++ -c --std=c++11 main.cxx -O3 $(CXXFLAGS)

clean:
	rm -f webcam-detector
