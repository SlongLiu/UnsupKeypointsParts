heatmap_size = (16, 16)
img_size = (128, 128)
batch_size = 64

seed = 123
nodes = 1

mixed_precision = True

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '9786'

# transformer
TRANS_CONTROLLER = dict(
    NAME = 'randTPStransform2',
    PARAS = dict(
        height = 128,
        width = 128,
    )
)

normalize_mean = [0.0] * 3
normalize_std = [1.0] * 3
data_transform_pipe = [
            dict(
                name = 'SLResize',
                paras = dict(size=(int(img_size[0]), int(img_size[1])))
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
            select_path = '/data/shilong/data/imm/celeba/MAFL/training.txt', # training
            anno_path = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = 0, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True, pic_trans_cont=False, soft_mask=False,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),

    TEST_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq' ,
            select_path = '/data/shilong/data/imm/celeba/MAFL/testing.txt', # training
            anno_path = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = 0, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True, soft_mask=False,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),
)

