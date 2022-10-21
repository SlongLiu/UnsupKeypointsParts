save_dir = None

img_size = (128, 128)
batch_size = 256

seed = 123
nodes = 1

lm_numb = 5 # number of landmarks to detection
subdivde = 4

mixed_precision = False 

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '9786'

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
            anno_path = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = False,
            saliency_dir = '/data/shilong/data/imm/celeba/parsing',
            predkp_dir = '/data/shilong/data/imm/celeba/predkp'
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = [0.0] * 3,
        normalize_std = [1.0] * 3,

        # for loader
        batch_size = batch_size,
        num_workers = 16,
        drop_last = True,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),

    TEST_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq',
            select_path = '/data/shilong/data/imm/celeba/MAFL/testing.txt',
            anno_path = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = False,
            saliency_dir = '/data/shilong/data/imm/celeba/parsing',
            predkp_dir = '/data/shilong/data/imm/celeba/predkp'
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = [0.0] * 3,
        normalize_std = [1.0] * 3,

        # for loader
        batch_size = batch_size,
        num_workers = 16,
        drop_last = True,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),

)

# model
MODEL = dict(
    NAME = 'MeshNetPlus',
    basename = 'MeshNetPlus',
    paras = dict(
        input_shape = img_size,
        nz_feat = 100,
        num_kps = 5,
        symmetric = True
    )
)

# loss
metriclist_train = [
    {
        'name': 'keyPointsLoss',
        'paras': {
            'item_map': dict(x='landmarks', y='output.kp_pred'),
        },
        'weight': 20.0
    },
    {
        'name': 'EntropyLoss',
        'paras': {
            'item_map': dict(x='output.vert2kp'),
        },
        'weight': 0.2
    },
    {
        'name': 'L2NormLoss',
        'paras': {
            'item_map': dict(x='output.delta_v'),
        },
        'weight': 10.0
    },
    {
        'name': 'LaplacianLoss',
        'paras': {
            'item_map': dict(verts='output.pred_v'),
        },
        'weight': 5.0
    },
    {
        'name': 'SaliencyLoss',
        'paras': {
            'item_map': dict(saliency='saliency', mask_pred='output.mask_pred'),
        },
        'weight': 1.0
    },
]

metriclist_test = None # the same as metriclist_train

# train
TRAIN = dict(
    epochs = 100,
    batch_size = batch_size,
    optimizer = 'Adam',
    lr = 1e-3,
    plot_interval = 1,
    test_interval = 1,

    print_interval_train = 50,
    weight_decay = 5e-4,

    regression_interval = 1,
    only_save_best_model = False
)


