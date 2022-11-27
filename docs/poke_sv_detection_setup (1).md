## poke deep yolox setup

## Param setting


```python
# train
work_dir = "work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600"
#resume_ckpt = "work_dirs/yolox_s_8x8_300e_PokeSVcoco/epoch_300.pth"
resume_ckpt = "work_dirs/yolox_s_8x8_300e_PokeSVcoco/latest.pth"
config_file = 'configs/yolox/yolox_s_8x8_300e_PokeSVcoco.py'

# inference
checkpoint_file = work_dir + '/epoch_300.pth'
```

### Google Drive


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
#%cd /content/drive/MyDrive/Ika
%cd /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection
```

    /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection



```python
!pip3 install openmim
!mim install mmcv-full

#!git clone -b feature/setup https://github.com/makiMakiTi/ika-ika-detection.git
#!git pull 

%cd ika-ika-detection
!pip install -e .
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: openmim in /usr/local/lib/python3.7/dist-packages (0.3.3)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from openmim) (0.8.10)
    Requirement already satisfied: colorama in /usr/local/lib/python3.7/dist-packages (from openmim) (0.4.6)
    Requirement already satisfied: model-index in /usr/local/lib/python3.7/dist-packages (from openmim) (0.1.11)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from openmim) (1.3.5)
    Requirement already satisfied: rich in /usr/local/lib/python3.7/dist-packages (from openmim) (12.6.0)
    Requirement already satisfied: Click in /usr/local/lib/python3.7/dist-packages (from openmim) (7.1.2)
    Requirement already satisfied: pip>=19.3 in /usr/local/lib/python3.7/dist-packages (from openmim) (21.1.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from openmim) (2.23.0)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from model-index->openmim) (6.0)
    Requirement already satisfied: markdown in /usr/local/lib/python3.7/dist-packages (from model-index->openmim) (3.4.1)
    Requirement already satisfied: ordered-set in /usr/local/lib/python3.7/dist-packages (from model-index->openmim) (4.1.0)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown->model-index->openmim) (4.13.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown->model-index->openmim) (3.10.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown->model-index->openmim) (4.1.1)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas->openmim) (1.21.6)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->openmim) (2022.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->openmim) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->openmim) (1.15.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->openmim) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->openmim) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->openmim) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->openmim) (2022.9.24)
    Requirement already satisfied: pygments<3.0.0,>=2.6.0 in /usr/local/lib/python3.7/dist-packages (from rich->openmim) (2.6.1)
    Requirement already satisfied: commonmark<0.10.0,>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from rich->openmim) (0.9.1)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Looking in links: https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
    Requirement already satisfied: mmcv-full in /usr/local/lib/python3.7/dist-packages (1.7.0)
    Requirement already satisfied: addict in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (2.4.0)
    Requirement already satisfied: yapf in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (0.32.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (21.3)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (6.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (1.21.6)
    Requirement already satisfied: opencv-python>=3 in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (4.6.0.66)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from mmcv-full) (7.1.2)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->mmcv-full) (3.0.9)
    [Errno 2] No such file or directory: 'ika-ika-detection'
    /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Obtaining file:///content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from mmdet==2.25.3) (3.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from mmdet==2.25.3) (1.21.6)
    Requirement already satisfied: pycocotools in /usr/local/lib/python3.7/dist-packages (from mmdet==2.25.3) (2.0.6)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from mmdet==2.25.3) (1.15.0)
    Requirement already satisfied: terminaltables in /usr/local/lib/python3.7/dist-packages (from mmdet==2.25.3) (3.1.10)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->mmdet==2.25.3) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->mmdet==2.25.3) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->mmdet==2.25.3) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->mmdet==2.25.3) (1.4.4)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->mmdet==2.25.3) (4.1.1)
    Installing collected packages: mmdet
      Attempting uninstall: mmdet
        Found existing installation: mmdet 2.25.3
        Can't uninstall 'mmdet'. No files were found to uninstall.
      Running setup.py develop for mmdet
    Successfully installed mmdet-2.25.3



```python

```

## download yolox model



```python
#!wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -P ./checkpoints/
```


```python
#%cp configs/yolox/yolox_s_8x8_300e_coco.py configs/yolox/yolox_l_8x8_300e_ika.py
```

## poke deep model train


```python
# !git pull 
```


```python

import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#!python tools/train.py $config_file --work-dir $work_dir  --resume-from $resume_ckpt
!python tools/train.py $config_file --work-dir $work_dir
#!python tools/train.py configs/yolox/yolox_s_8x8_300e_coco_ika3.py
```

    /usr/local/lib/python3.7/dist-packages/mmcv/__init__.py:21: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
      'On January 1, 2023, MMCV will release v2.0.0, in which it will remove '
    /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/mmdet/utils/setup_env.py:39: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
      f'Setting OMP_NUM_THREADS environment variable for each process '
    /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/mmdet/utils/setup_env.py:49: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
      f'Setting MKL_NUM_THREADS environment variable for each process '
    2022-11-21 09:22:44,909 - mmdet - INFO - Environment info:
    ------------------------------------------------------------
    sys.platform: linux
    Python: 3.7.15 (default, Oct 12 2022, 19:14:55) [GCC 7.5.0]
    CUDA available: True
    GPU 0: Tesla T4
    CUDA_HOME: /usr/local/cuda
    NVCC: Cuda compilation tools, release 11.2, V11.2.152
    GCC: x86_64-linux-gnu-gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
    PyTorch: 1.12.1+cu113
    PyTorch compiling details: PyTorch built with:
      - GCC 9.3
      - C++ Version: 201402
      - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
      - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
      - OpenMP 201511 (a.k.a. OpenMP 4.5)
      - LAPACK is enabled (usually provided by MKL)
      - NNPACK is enabled
      - CPU capability usage: AVX2
      - CUDA Runtime 11.3
      - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
      - CuDNN 8.3.2  (built against CUDA 11.5)
      - Magma 2.5.2
      - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.3.2, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.12.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 
    
    TorchVision: 0.13.1+cu113
    OpenCV: 4.6.0
    MMCV: 1.7.0
    MMCV Compiler: GCC 9.3
    MMCV CUDA Compiler: 11.3
    MMDetection: 2.25.3+6019f49
    ------------------------------------------------------------
    
    2022-11-21 09:22:46,731 - mmdet - INFO - Distributed training: False
    2022-11-21 09:22:49,635 - mmdet - INFO - Config:
    optimizer = dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
        paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
    optimizer_config = dict(grad_clip=None)
    lr_config = dict(
        policy='YOLOX',
        warmup='exp',
        by_epoch=False,
        warmup_by_epoch=True,
        warmup_ratio=1,
        warmup_iters=5,
        num_last_epochs=15,
        min_lr_ratio=0.05)
    runner = dict(type='EpochBasedRunner', max_epochs=500)
    checkpoint_config = dict(interval=10)
    log_config = dict(interval=50, hooks=[dict(type='TensorboardLoggerHook')])
    custom_hooks = [
        dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
        dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48)
    ]
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    load_from = 'work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1_300/epoch_300.pth'
    resume_from = None
    workflow = [('train', 1)]
    opencv_num_threads = 0
    mp_start_method = 'fork'
    auto_scale_lr = dict(enable=False, base_batch_size=64)
    img_scale = (640, 640)
    classes = ('Delvil', 'Digda', 'Gourton', 'Hanecco', 'Hellgar', 'Hogator',
               'Kofukimushi', 'Koraidon', 'Kuwassu', 'Nyahoja', 'Tarountula',
               'Yayakoma', 'Youngoose', 'player')
    model = dict(
        type='YOLOX',
        input_size=(640, 640),
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=1.0, widen_factor=1.0),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[256, 512, 1024],
            out_channels=256,
            num_csp_blocks=3),
        bbox_head=dict(
            type='YOLOXHead', num_classes=14, in_channels=256, feat_channels=256),
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
    data_root = '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/'
    dataset_type = 'CocoDataset'
    train_pipeline = [
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine', scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    train_dataset = dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=('Delvil', 'Digda', 'Gourton', 'Hanecco', 'Hellgar', 'Hogator',
                     'Kofukimushi', 'Koraidon', 'Kuwassu', 'Nyahoja', 'Tarountula',
                     'Yayakoma', 'Youngoose', 'player'),
            ann_file=
            '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/pokemon_sv_train.json',
            img_prefix=
            '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ])
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(640, 640),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Pad',
                    pad_to_square=True,
                    pad_val=dict(img=(114.0, 114.0, 114.0))),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ])
    ]
    data = dict(
        samples_per_gpu=3,
        workers_per_gpu=3,
        persistent_workers=True,
        train=dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type='CocoDataset',
                classes=('Delvil', 'Digda', 'Gourton', 'Hanecco', 'Hellgar',
                         'Hogator', 'Kofukimushi', 'Koraidon', 'Kuwassu',
                         'Nyahoja', 'Tarountula', 'Yayakoma', 'Youngoose',
                         'player'),
                ann_file=
                '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/pokemon_sv_train.json',
                img_prefix=
                '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/train2017/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                filter_empty_gt=False),
            pipeline=[
                dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
                dict(
                    type='RandomAffine',
                    scaling_ratio_range=(0.1, 2),
                    border=(-320, -320)),
                dict(
                    type='MixUp',
                    img_scale=(640, 640),
                    ratio_range=(0.8, 1.6),
                    pad_val=114.0),
                dict(type='YOLOXHSVRandomAug'),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
                dict(
                    type='Pad',
                    pad_to_square=True,
                    pad_val=dict(img=(114.0, 114.0, 114.0))),
                dict(
                    type='FilterAnnotations',
                    min_gt_bbox_wh=(1, 1),
                    keep_empty=False),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        val=dict(
            type='CocoDataset',
            classes=('Delvil', 'Digda', 'Gourton', 'Hanecco', 'Hellgar', 'Hogator',
                     'Kofukimushi', 'Koraidon', 'Kuwassu', 'Nyahoja', 'Tarountula',
                     'Yayakoma', 'Youngoose', 'player'),
            ann_file=
            '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/pokemon_sv_valid.json',
            img_prefix=
            '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/val2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 640),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(
                            type='Pad',
                            pad_to_square=True,
                            pad_val=dict(img=(114.0, 114.0, 114.0))),
                        dict(type='DefaultFormatBundle'),
                        dict(type='Collect', keys=['img'])
                    ])
            ]),
        test=dict(
            type='CocoDataset',
            classes=('Delvil', 'Digda', 'Gourton', 'Hanecco', 'Hellgar', 'Hogator',
                     'Kofukimushi', 'Koraidon', 'Kuwassu', 'Nyahoja', 'Tarountula',
                     'Yayakoma', 'Youngoose', 'player'),
            ann_file=
            '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/pokemon_sv_valid.json',
            img_prefix=
            '/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v1.0/val2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 640),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(
                            type='Pad',
                            pad_to_square=True,
                            pad_val=dict(img=(114.0, 114.0, 114.0))),
                        dict(type='DefaultFormatBundle'),
                        dict(type='Collect', keys=['img'])
                    ])
            ]))
    max_epochs = 500
    num_last_epochs = 15
    interval = 10
    evaluation = dict(
        save_best='auto', interval=10, dynamic_intervals=[(485, 1)], metric='bbox')
    work_dir = 'work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600'
    auto_resume = False
    gpu_ids = [0]
    
    2022-11-21 09:22:49,636 - mmdet - INFO - Set random seed to 1355098491, deterministic: False
    2022-11-21 09:22:51,363 - mmdet - INFO - initialize CSPDarknet with init_cfg {'type': 'Kaiming', 'layer': 'Conv2d', 'a': 2.23606797749979, 'distribution': 'uniform', 'mode': 'fan_in', 'nonlinearity': 'leaky_relu'}
    2022-11-21 09:22:51,964 - mmdet - INFO - initialize YOLOXPAFPN with init_cfg {'type': 'Kaiming', 'layer': 'Conv2d', 'a': 2.23606797749979, 'distribution': 'uniform', 'mode': 'fan_in', 'nonlinearity': 'leaky_relu'}
    2022-11-21 09:22:52,372 - mmdet - INFO - initialize YOLOXHead with init_cfg {'type': 'Kaiming', 'layer': 'Conv2d', 'a': 2.23606797749979, 'distribution': 'uniform', 'mode': 'fan_in', 'nonlinearity': 'leaky_relu'}
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:566: UserWarning: This DataLoader will create 3 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    2022-11-21 09:22:55,710 - mmdet - INFO - Automatic scaling of learning rate (LR) has been disabled.
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    2022-11-21 09:22:55,760 - mmdet - INFO - load checkpoint from local path: work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1_300/epoch_300.pth
    2022-11-21 09:22:56,637 - mmdet - WARNING - The model and loaded state dict do not match exactly
    
    size mismatch for bbox_head.multi_level_conv_cls.0.weight: copying a param with shape torch.Size([4, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([14, 256, 1, 1]).
    size mismatch for bbox_head.multi_level_conv_cls.0.bias: copying a param with shape torch.Size([4]) from checkpoint, the shape in current model is torch.Size([14]).
    size mismatch for bbox_head.multi_level_conv_cls.1.weight: copying a param with shape torch.Size([4, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([14, 256, 1, 1]).
    size mismatch for bbox_head.multi_level_conv_cls.1.bias: copying a param with shape torch.Size([4]) from checkpoint, the shape in current model is torch.Size([14]).
    size mismatch for bbox_head.multi_level_conv_cls.2.weight: copying a param with shape torch.Size([4, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([14, 256, 1, 1]).
    size mismatch for bbox_head.multi_level_conv_cls.2.bias: copying a param with shape torch.Size([4]) from checkpoint, the shape in current model is torch.Size([14]).
    2022-11-21 09:22:56,657 - mmdet - INFO - Start running, host: root@dbcbefdc5e6c, work_dir: /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600
    2022-11-21 09:22:56,658 - mmdet - INFO - Hooks will be executed in the following order:
    before_run:
    (VERY_HIGH   ) YOLOXLrUpdaterHook                 
    (NORMAL      ) CheckpointHook                     
    (LOW         ) EvalHook                           
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    before_train_epoch:
    (VERY_HIGH   ) YOLOXLrUpdaterHook                 
    (48          ) YOLOXModeSwitchHook                
    (48          ) SyncNormHook                       
    (LOW         ) IterTimerHook                      
    (LOW         ) EvalHook                           
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    before_train_iter:
    (VERY_HIGH   ) YOLOXLrUpdaterHook                 
    (LOW         ) IterTimerHook                      
    (LOW         ) EvalHook                           
     -------------------- 
    after_train_iter:
    (ABOVE_NORMAL) OptimizerHook                      
    (NORMAL      ) CheckpointHook                     
    (LOW         ) IterTimerHook                      
    (LOW         ) EvalHook                           
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    after_train_epoch:
    (48          ) SyncNormHook                       
    (NORMAL      ) CheckpointHook                     
    (LOW         ) EvalHook                           
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    before_val_epoch:
    (LOW         ) IterTimerHook                      
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    before_val_iter:
    (LOW         ) IterTimerHook                      
     -------------------- 
    after_val_iter:
    (LOW         ) IterTimerHook                      
     -------------------- 
    after_val_epoch:
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    after_run:
    (VERY_LOW    ) TensorboardLoggerHook              
     -------------------- 
    2022-11-21 09:22:56,658 - mmdet - INFO - workflow: [('train', 1)], max: 500 epochs
    2022-11-21 09:22:56,658 - mmdet - INFO - Checkpoints will be saved to /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600 by HardDiskBackend.
    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:566: UserWarning: This DataLoader will create 3 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    /usr/local/lib/python3.7/dist-packages/torch/functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2894.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    2022-11-21 09:36:00,016 - mmdet - INFO - Saving checkpoint at 10 epochs
    INFO:mmdet:Saving checkpoint at 10 epochs
    [>>] 118/118, 10.5 task/s, elapsed: 11s, ETA:     0s2022-11-21 09:36:14,618 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.31s).
    Accumulating evaluation results...
    DONE (t=0.19s).
    2022-11-21 09:36:15,460 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.014
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.030
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.010
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.019
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.056
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.056
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.056
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.006
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.071
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.014
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.030
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.010
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.019
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.056
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.056
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.056
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.006
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.071
    
    2022-11-21 09:36:18,450 - mmdet - INFO - Now best checkpoint is saved as best_bbox_mAP_epoch_10.pth.
    INFO:mmdet:Now best checkpoint is saved as best_bbox_mAP_epoch_10.pth.
    2022-11-21 09:36:18,461 - mmdet - INFO - Best bbox_mAP is 0.0140 at 10 epoch.
    INFO:mmdet:Best bbox_mAP is 0.0140 at 10 epoch.
    2022-11-21 09:49:03,656 - mmdet - INFO - Saving checkpoint at 20 epochs
    INFO:mmdet:Saving checkpoint at 20 epochs
    [>>] 118/118, 9.0 task/s, elapsed: 13s, ETA:     0s2022-11-21 09:49:19,608 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.58s).
    Accumulating evaluation results...
    DONE (t=0.23s).
    2022-11-21 09:49:20,495 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.014
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.038
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.006
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.025
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.020
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.025
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.011
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.146
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.014
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.038
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.006
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.025
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.020
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.025
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.011
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.146
    
    2022-11-21 10:02:28,776 - mmdet - INFO - Saving checkpoint at 30 epochs
    INFO:mmdet:Saving checkpoint at 30 epochs
    [>>] 118/118, 9.6 task/s, elapsed: 12s, ETA:     0s2022-11-21 10:02:44,236 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.44s).
    Accumulating evaluation results...
    DONE (t=0.24s).
    2022-11-21 10:02:44,991 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.041
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.086
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.031
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.002
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.111
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.026
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.222
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.041
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.086
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.031
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.002
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.111
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.026
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.222
    
    2022-11-21 10:02:45,007 - mmdet - INFO - The previous best checkpoint /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600/best_bbox_mAP_epoch_10.pth was removed
    INFO:mmdet:The previous best checkpoint /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600/best_bbox_mAP_epoch_10.pth was removed
    2022-11-21 10:02:47,676 - mmdet - INFO - Now best checkpoint is saved as best_bbox_mAP_epoch_30.pth.
    INFO:mmdet:Now best checkpoint is saved as best_bbox_mAP_epoch_30.pth.
    2022-11-21 10:02:47,676 - mmdet - INFO - Best bbox_mAP is 0.0410 at 30 epoch.
    INFO:mmdet:Best bbox_mAP is 0.0410 at 30 epoch.
    2022-11-21 10:16:01,544 - mmdet - INFO - Saving checkpoint at 40 epochs
    INFO:mmdet:Saving checkpoint at 40 epochs
    [>>] 118/118, 9.3 task/s, elapsed: 13s, ETA:     0s2022-11-21 10:16:17,460 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.34s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.75s).
    Accumulating evaluation results...
    DONE (t=0.42s).
    2022-11-21 10:16:19,050 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.018
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.012
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.002
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.022
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.092
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.092
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.006
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.114
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.018
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.012
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.002
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.022
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.092
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.092
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.006
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.114
    
    2022-11-21 10:29:28,514 - mmdet - INFO - Saving checkpoint at 50 epochs
    INFO:mmdet:Saving checkpoint at 50 epochs
    [>>] 118/118, 10.0 task/s, elapsed: 12s, ETA:     0s2022-11-21 10:29:43,452 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.25s).
    Accumulating evaluation results...
    DONE (t=0.21s).
    2022-11-21 10:29:43,952 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.030
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.058
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.025
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.005
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.039
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.099
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.099
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.099
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.005
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.132
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.030
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.058
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.025
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.005
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.039
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.099
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.099
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.099
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.005
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.132
    
    2022-11-21 10:43:04,234 - mmdet - INFO - Saving checkpoint at 60 epochs
    INFO:mmdet:Saving checkpoint at 60 epochs
    [>>] 118/118, 10.2 task/s, elapsed: 12s, ETA:     0s2022-11-21 10:43:19,018 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.40s).
    Accumulating evaluation results...
    DONE (t=0.26s).
    2022-11-21 10:43:19,938 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.025
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.060
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.014
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.018
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.027
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.094
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.094
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.094
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.031
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.113
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.025
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.060
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.014
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.018
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.027
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.094
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.094
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.094
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.031
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.113
    
    2022-11-21 10:56:19,590 - mmdet - INFO - Saving checkpoint at 70 epochs
    INFO:mmdet:Saving checkpoint at 70 epochs
    [>>] 118/118, 9.6 task/s, elapsed: 12s, ETA:     0s2022-11-21 10:56:34,829 - mmdet - INFO - Evaluating bbox...
    INFO:mmdet:Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.01s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.24s).
    Accumulating evaluation results...
    DONE (t=0.18s).
    2022-11-21 10:56:35,281 - mmdet - INFO - 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.044
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.118
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.020
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.052
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.066
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.167
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.167
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.167
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.066
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.273
    
    INFO:mmdet:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.044
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.118
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.020
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.052
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.066
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.167
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.167
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.167
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.066
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.273
    
    2022-11-21 10:56:35,291 - mmdet - INFO - The previous best checkpoint /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600/best_bbox_mAP_epoch_30.pth was removed
    INFO:mmdet:The previous best checkpoint /content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Detection/mmdetection/work_dirs/yolox_s_8x8_300e_PokeSVcoco_v1.0_600/best_bbox_mAP_epoch_30.pth was removed
    2022-11-21 10:56:38,119 - mmdet - INFO - Now best checkpoint is saved as best_bbox_mAP_epoch_70.pth.
    INFO:mmdet:Now best checkpoint is saved as best_bbox_mAP_epoch_70.pth.
    2022-11-21 10:56:38,119 - mmdet - INFO - Best bbox_mAP is 0.0440 at 70 epoch.
    INFO:mmdet:Best bbox_mAP is 0.0440 at 70 epoch.



```python
!nvidia-smi
```

## Inference


```python
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
```


```python

#config_file = 'work_dirs/yolox_s_8x8_300e_coco_ika3/yolox_s_8x8_300e_coco_ika3.py'

#checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco_ika3/latest.pth'


#checkpoint_file = '/content/drive/MyDrive/Ika/ika-ika-detection/work_dirs/yolox_s_8x8_300e_coco_ika3/epoch_300.pth'
#checkpoint_file = 'checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'

device = 'cuda:0'
# device = 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)
```


```python
import pathlib

for f in pathlib.Path('/content/drive/MyDrive/PROJECT/201_HaMaruki/201_60_PokemonSV/Pokemon-SV-Datasets/datasets/v0/val2017').glob('*jpg'):

    result = inference_detector(model, f)

    model.show_result(f,
        result, 
        out_file=pathlib.Path(work_dir + '/inference/valid') / f.name
    )
```




