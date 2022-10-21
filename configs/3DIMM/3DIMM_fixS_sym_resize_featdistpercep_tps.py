save_dir = None

img_size = (128, 128)
batch_size = 160

seed = 123
nodes = 1

lm_numb =  68 # number of landmarks to detection
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
            pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_trans_cont = False, pic_return_mask = True
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
            root_dir = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq',
            select_path = '/data/shilong/data/imm/celeba/MAFL/testing.txt',
            anno_path = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_trans_cont = False, pic_return_mask = True
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

# model
MODEL = dict(
    NAME = 'IMM3D',
    basename = 'IMM3D',
    paras = dict(
        nz_feat=128, 
        n_kp=68, 
        input_shape=(128, 128), 
        mean_v_path='tmp/ave_lm_norm_resize.npy', 
        mean_v_fix=True,
        heatmap_size=(16, 16), 
        heatmap_std=0.02, 
        heatmap_mode='point', 
        h_channel=32
    )
)

# loss
metriclist_train = [
    {
        'name': 'PerceptualLoss',
        'paras': {
            'model_type': 'Vgg19',
            'content_layer': [0, 1, 2, 3, 4, 5],
            'loss_type': 'L2',
            'layer_weight': [1/0.25, 1/0.26, 1/1.4, 1/3.19, 1/2.54, 1/0.3],
            'item_map': [dict(x='img1', y='output.recovered_1', mask='img1_mask'),
                        dict(x='img1', y='output.recovered_1_sym', mask='img1_mask')],
            'refresh_iterval': 100,
        },
        'weight': 1.0
    },
    {
        'name': 'FeatDistPercepLoss',
        'paras': {
            'fix_feat_path': 'tmp/ave_feat_10000.npy',
            'item_map': dict(featlist='output.local_feat'),
        },
        'weight': 0.1
    },
    {
        'name': 'L2NormLoss',
        'paras': {
            'item_map': dict(x='output.delta_v1'),
        },
        'weight': 100.0
    },

]

metriclist_test = None # the same as metriclist_train

SCHEDULER = dict(
    name = 'StepLR',
    paras = dict(
        step_size = 15,
        gamma = 0.3
    )
)

# train
TRAIN = dict(
    epochs = 60,
    batch_size = batch_size,
    optimizer = 'Adam',
    lr = 1e-4,
    plot_interval = 1,
    test_interval = 1,

    print_interval_train = 50,
    weight_decay = 5e-4,

    only_save_best_model = True,
)

