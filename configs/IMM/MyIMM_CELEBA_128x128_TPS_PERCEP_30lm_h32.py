save_dir = None

heatmap_size = (16, 16)
img_size = (128, 128)
batch_size = 128

seed = 123
nodes = 1

lm_numb = 30 # number of landmarks to detection
mixed_precision = True # there's a bug if setting as True

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '9786'

# transformer
TRANS_CONTROLLER = dict(
    NAME = 'randTPSTransform',
    PARAS = dict(
        num_control = 5,
        tps_variance = 0.08,
        rotate_variance = 0.05
    )
)

# dataset
DATASET = dict(
    TRAIN_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba_sl/Img/img_align_celeba_hq' ,
            select_path = '/data/shilong/data/imm/celeba_sl/MAFL/all_except_testing.txt', # training
            anno_path = '/data/shilong/data/imm/celeba_sl/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = [0.0] * 3,
        normalize_std = [1.0] * 3,

        # for loader
        batch_size = batch_size,
        num_workers = 16,
    ),

    TEST_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba_sl/Img/img_align_celeba_hq',
            select_path = '/data/shilong/data/imm/celeba_sl/MAFL/testing.txt',
            anno_path = '/data/shilong/data/imm/celeba_sl/Anno/list_landmarks_align_celeba_normed.txt',
            pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = [0.0] * 3,
        normalize_std = [1.0] * 3,

        # for loader
        batch_size = batch_size,
        num_workers = 16,
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
            'layer_weight': [1/0.5, 1/0.26, 1/1.57, 1/3.19, 1/2.54, 1/0.21],
            'item_map': dict(x='img2', y='output.recovered_y', mask='img2_mask')
        },
        'weight': 1.0
    }
]

metriclist_test = None # the same as metriclist_train

# train
TRAIN = dict(
    epochs = 30,
    batch_size = batch_size,
    optimizer = 'Adam',
    lr = 1e-4,
    plot_interval = 1,
    test_interval = 1,

    print_interval_train = 50,
    weight_decay = 5e-4,

    regression_interval = 1,
    only_save_best_model = True
)



###############################
######### regression ##########
###############################

REGRESSION_MAFL = dict(

    MODELNAME = MODEL['NAME'],

    DATASET = dict(
        TRAIN_SET = dict(
            # for dataset
            dataset = 'celeba',
            paras = dict(
                root_dir = '/data/shilong/data/imm/celeba_sl/Img/img_align_celeba_hq',
                select_path = '/data/shilong/data/imm/celeba_sl/MAFL/training.txt', # training
                anno_path = '/data/shilong/data/imm/celeba_sl/Anno/list_landmarks_align_celeba_normed.txt',
                pic_trans_num = 0
            ),

            data_rescale_height = img_size[0],
            data_rescale_width = img_size[1],
            normalize_mean = [0.0] * 3,
            normalize_std = [1.0] * 3,

            # for loader
            batch_size = batch_size,
            drop_last_train = False,
            num_workers = 16,
        ),

        TEST_SET = dict(
            # for dataset
            dataset = 'celeba',
            paras = dict(
                root_dir = '/data/shilong/data/imm/celeba_sl/Img/img_align_celeba_hq',
                select_path = '/data/shilong/data/imm/celeba_sl/MAFL/testing.txt',
                anno_path = '/data/shilong/data/imm/celeba_sl/Anno/list_landmarks_align_celeba_normed.txt',
                pic_trans_num = 0
            ),

            data_rescale_height = img_size[0],
            data_rescale_width = img_size[1],
            normalize_mean = [0.0] * 3,
            normalize_std = [1.0] * 3,

            # for loader
            batch_size = batch_size,
            drop_last_train = False,
            num_workers = 16,
        ),
    )
)


# regression
REGRESSION_AFLW = dict(

    MODELNAME = MODEL['NAME'],

    DATASET = dict(
        TRAIN_SET = dict(
            # for dataset
            dataset = 'celeba',
            paras = dict(
                root_dir = '/data/shilong/data/imm/aflw_sl/Image',
                select_path = '/data/shilong/data/imm/aflw_sl/split/train.txt', # training
                anno_path = '/data/shilong/data/imm/aflw_sl/anno/anno.txt',
                pic_trans_num = 0
            ),

            data_rescale_height = img_size[0],
            data_rescale_width = img_size[1],
            normalize_mean = [0.0] * 3,
            normalize_std = [1.0] * 3,
            
            # for loader
            batch_size = batch_size,
            drop_last_train = False,
            num_workers = 16,
        ),

        TEST_SET = dict(
            # for dataset
            dataset = 'celeba',
            paras = dict(
                root_dir = '/data/shilong/data/imm/aflw_sl/Image',
                select_path = '/data/shilong/data/imm/aflw_sl/split/test.txt', # training
                anno_path = '/data/shilong/data/imm/aflw_sl/anno/anno.txt',
                pic_trans_num = 0
            ),

            data_rescale_height = img_size[0],
            data_rescale_width = img_size[1],
            normalize_mean = [0.0] * 3,
            normalize_std = [1.0] * 3,

            # for loader
            batch_size = batch_size,
            drop_last_train = False,
            num_workers = 16,
        ),
    ),
)