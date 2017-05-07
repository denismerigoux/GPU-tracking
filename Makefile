BUILD_PATH := /afs/andrew.cmu.edu/usr14/ideutel/builds-15618-project
# LD_LIBRARY_PATH := $LD_LIBRARY_PATH:${BUILD_PATH}/lib:/usr/lib64/:/usr/lib64/nvidia/

ghc: src/main.cpp src/trackerKCFparallel.hpp src/trackerKCFparallel.cu CMakeLists.txt
	mkdir -p build
	cd build && \
	cmake \
	    -DCMAKE_PREFIX_PATH=${BUILD_PATH}/share/OpenCV \
		  -DOpenCV2_INCLUDE_PATH=${BUILD_PATH}/include/opencv2 \
			-DCMAKE_BUILD_TYPE=RELEASE .. && \
	make
	ln -sf build/main main

clean:
	rm -rf build main
