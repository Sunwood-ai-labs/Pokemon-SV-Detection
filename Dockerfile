FROM nvcr.io/nvidia/pytorch:22.10-py3

# -------------------------------------
# init
#
RUN python3 -m pip install --upgrade pip
WORKDIR /home
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1

# -------------------------------------
# mmcv
#
RUN pip install -U openmim
# RUN pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.13.0/index.html
RUN pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
# RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
# RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
# RUN pip install mmdet
RUN mim install mmdet
RUN pip install opencv-python==4.5.5.64

# -------------------------------------
# MMDetection
#
RUN git clone https://github.com/open-mmlab/mmdetection.git ; exit 0
RUN cd mmdetection && pip install -v -e .

RUN pip install redis
RUN pip install rq
RUN pip install boto3

# -------------------------------------
# notebook
#
RUN pip install notebook
