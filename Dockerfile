#FROM ubuntu:20.04
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get upgrade -qq

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq
RUN apt-get install -y  \
  build-essential \
  cmake \
  git \
  graphviz \
  libatlas-base-dev \
  libboost-filesystem-dev \
  libboost-iostreams-dev \
  libboost-program-options-dev \
  libboost-regex-dev \
  libboost-serialization-dev \
  libboost-system-dev \
  libboost-test-dev \
  libboost-graph-dev \
  libcgal-dev \
  libcgal-qt5-dev \
  libfreeimage-dev \
  libgflags-dev \
  libglew-dev \
  libglu1-mesa-dev \
  libgoogle-glog-dev \
  libjpeg-dev \
  libopencv-dev \
  libpng-dev \
  libqt5opengl5-dev \
  libsuitesparse-dev \
  libtiff-dev \
  libxi-dev \
  libxrandr-dev \
  libxxf86vm-dev \
  libxxf86vm1 \
  mediainfo \
  mercurial \
  qtbase5-dev \
  libatlas-base-dev \
  libsuitesparse-dev

RUN apt-get install -y  \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libglew-dev \
  qtbase5-dev \
  libcgal-dev


WORKDIR /tmp/build

# Install openmvg
RUN git clone -b develop --recursive https://github.com/openMVG/openMVG.git openmvg && \
  mkdir openmvg_build && cd openmvg_build && \
  cmake -DCMAKE_BUILD_TYPE=RELEASE . ../openmvg/src -DCMAKE_INSTALL_PREFIX=/opt/openmvg && \
  make -j18  && \
  make install

# Install eigen
RUN git clone https://gitlab.com/libeigen/eigen.git --branch 3.4 && \
  mkdir eigen_build && cd eigen_build && \
  cmake . ../eigen && \
  make -j18 && \
  make install 

# Get vcglib
RUN git clone https://github.com/cdcseacave/VCG.git vcglib 

# Install ceres solver
RUN git clone https://ceres-solver.googlesource.com/ceres-solver ceres_solver && \
  cd ceres_solver && git checkout tags/1.14.0 && cd .. && \
  mkdir ceres_build && cd ceres_build && \
  cmake . ../ceres_solver/ -DMINIGLOG=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
  make -j18 && \
  make install

RUN apt-get update && apt-get install -y libglfw3 libglfw3-dev
# Install openmvs
RUN git clone https://github.com/cdcseacave/openMVS.git openmvs && \
  mkdir openmvs_build && cd openmvs_build && \
  cmake . ../openmvs -DOpenMVS_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DVCG_DIR="../vcglib" -DCMAKE_INSTALL_PREFIX=/opt/openmvs && \
  make -j18 && \
  make install

# Install cmvs-pmvs
RUN git clone https://github.com/pmoulon/CMVS-PMVS /tmp/build/cmvs-pmvs && \
  mkdir /tmp/build/cmvs-pmvs_build && cd /tmp/build/cmvs-pmvs_build && \
  cmake ../cmvs-pmvs/program -DCMAKE_INSTALL_PREFIX=/opt/cmvs && \
  make -j18 && \
  make install

RUN apt-get update && apt-get install -y \
  git \
  cmake \
  build-essential \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libboost-test-dev \
  libeigen3-dev \
  libsuitesparse-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  qt5-default 


RUN apt install ubuntu-drivers-common -y
RUN ubuntu-drivers autoinstall


# Install colmap
# Both master and dev seem broken so commented out for now
RUN git clone -b dev https://github.com/colmap/colmap /tmp/build/colmap && \
  mkdir -p /tmp/build/colmap_build && cd /tmp/build/colmap_build && \
  cmake . ../colmap -DCMAKE_INSTALL_PREFIX=/opt/colmap -DGUI_ENABLE=FALSE -DCUDA_ENABLE=FALSE -DQT_QPA_PLATFORM=offscreen && \
  #sudo cmake . ../colmap -DCMAKE_INSTALL_PREFIX=/opt/colmap -DGUI_ENABLE=FALSE -DCUDA_ENABLE=FALSE -DQT_QPA_PLATFORM=offscreen -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  make -j18 && \
  make install

# ..and then create a more lightweight image to actually run stuff in.
#FROM updated
ENV DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000
RUN apt-get update && apt-get install -y \
  curl \
  exiftool \
  ffmpeg \
  mediainfo \
  graphviz \
  libpng16-16 \
  libjpeg-turbo8 \
  libtiff5 \
  libxxf86vm1 \
  libxi6 \
  libxrandr2 \
  libatlas-base-dev \
  libqt5widgets5 \
  libboost-iostreams1.71.0 \
  libboost-program-options1.71.0 \
  libboost-serialization1.71.0 \
  libopencv-calib3d4.2 \
  libopencv-highgui4.2 \
  libgoogle-glog0v5 \
  libfreeimage3 \
  libcgal-dev \
  libglew2.1 \
  libcholmod3 \
  libcxsparse3 \
  python2-minimal \
  git \
  cmake \
  build-essential \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libboost-test-dev \
  libeigen3-dev \
  libsuitesparse-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  qt5-default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2 1
#COPY --from=build /opt /opt
COPY pipeline.py /opt/dpg/pipeline.py
COPY COLMAP_MVS_pipeline.py /opt/dpg/colmap_mvs_pipeline.py
RUN echo ptools soft core unlimited >> /etc/security/limits.conf
RUN echo ptools hard core unlimited >> /etc/security/limits.conf
RUN groupadd -g $GID ptools
RUN useradd -r -u $UID -m -g ptools ptools
WORKDIR /
USER ptools
ENV PATH=/opt/openmvs/bin/OpenMVS:/opt/openmvg/bin:/opt/cmvs/bin:/opt/colmap/bin:/opt/dpg:$PATH
