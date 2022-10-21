heatmap_size = (16, 16)
img_size = (128, 128)
batch_size = 128

seed = 123
nodes = 1

lm_numb = 10 # number of landmarks to detection
mixed_precision = True

savepath = '/data/shilong/data/imm/celeba/2d_lm_pred/lm_pred_ep0.json'
resume = 'log/0912/IMM_celeba_v8_10lm_0.02hm_tps2_percept_h32_r100_1x_lr1e-4_point_softmask_2/models/ep38.pkl'

normalize_mean = [0.0] * 3
normalize_std = [1.0] * 3
data_transform_pipe = [
            dict(
                name = 'SLResize',
                paras = dict(size=(int(img_size[0]/0.8),int(img_size[1]/0.8)))
            ),
            dict(
                name = 'SLCenterCrop',
                paras = dict(size=img_size)
            ),
            dict(name='SLToTensor', paras={}),
            dict(
                name = 'SLNormalize',
                paras = dict(mean=normalize_mean, std=normalize_std)
            ),
        ]

# dataset
DATASET = dict(
    TRAIN_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq' ,
            select_path = '/data/shilong/data/imm/celeba/MAFL/all_except_testing.txt', # training
            anno_path = None,
            pic_trans_num = 0
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,
        drop_last = False,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),

    TEST_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq',
            select_path = None,
            anno_path = None,
            pic_trans_num = 0
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,
        drop_last = False,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),
)

# model
MODEL = dict(
    NAME = 'MyIMM',
    basename = 'MyIMM',
    paras = dict(
        dim = lm_numb,
        in_channel = 3,
        h_channel = 32,
        mode = 'point',
        heatmap_std = 0.02,
    )
)

metriclist_train = []