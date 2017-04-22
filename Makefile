build: src/main.cpp
	mkdir -p build
	cd build && \
	cmake -D OpenCV_DIR=/tmp/dmerigou/opencv-3.2.0/build2 -DCMAKE_BUILD_TYPE=RELEASE .. && \
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/dmerigou/ffmpeg-3.2.4/build/lib make
	ln -sf build/main main
