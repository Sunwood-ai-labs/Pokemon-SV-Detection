from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import pathlib


config_file = "configs/yolox/yolox_s_8x8_300e_PokeSVcoco.py"
work_dir = "/home/pokemon-sv-work_dirs/yolox_s_8x8_300e_PokeSVcoco_v2.2.8_0300"
checkpoint_file = work_dir + "/epoch_300.pth"
vaild_image_path = "/home/pokemon-sv-datasets/datasets/v2.2/val2017"

device = 'cuda:0'
# device = 'cpu'

model = init_detector(config_file, checkpoint_file, device=device)



for f in pathlib.Path(vaild_image_path).glob('*jpg'):

    result = inference_detector(model, f)

    model.show_result(f,
        result, 
        out_file=pathlib.Path(work_dir + '/inference/valid') / f.name
    )