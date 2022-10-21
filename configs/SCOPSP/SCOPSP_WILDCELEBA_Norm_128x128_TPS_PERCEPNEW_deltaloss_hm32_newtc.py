save_dir = None

heatmap_size = (64, 64)
img_size = (256, 256)
batch_size = 8

seed = 123
nodes = 1

lm_numb = 9 # number of landmarks to detection
mixed_precision = False
reg_list = 'wildceleba'

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '9786'

# transformer
TRANS_CONTROLLER = dict(
    NAME = 'randTPStransform2',
    PARAS = dict(
        height = 256,
        width = 256,
    )
)

normalize_mean = [0.0] * 3
normalize_std = [1.0] * 3
data_transform_pipe = [
            dict(
                name = 'Resize',
                paras = dict(size=(int(img_size[0]),int(img_size[1])))
            ),
            dict(name='ToTensor', paras={}),
            dict(
                name = 'Normalize',
                paras = dict(mean=normalize_mean, std=normalize_std)
            ),
        ]

# dataset
DATASET = dict(
    TRAIN_SET = dict(
        # for dataset
        dataset = 'wildceleba',
        paras = dict(
            root_dir = '/data/shilong/data/wildceleba/Img' ,
            select_path = '/data/shilong/data/wildceleba/MAFL_useful/all_wo_train_test.txt', # training
            anno_path = '/data/shilong/data/wildceleba/anno/list_landmarks_celeba_normed.txt',
            pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True, pic_trans_cont=False, soft_mask=False,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,

        # for pipeline
        Tname = 'T',
        pipeline = data_transform_pipe
    ),

    TEST_SET = dict(
        # for dataset
        dataset = 'wildceleba',
        paras = dict(
            root_dir = '/data/shilong/data/wildceleba/Img',
            select_path = '/data/shilong/data/wildceleba/MAFL_useful/testing.txt',
            anno_path = '/data/shilong/data/wildceleba/anno/list_landmarks_celeba_normed.txt',
            pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True, soft_mask=False,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,

        # for pipeline
        Tname = 'T',
        pipeline = data_transform_pipe
    ),
)

# model
MODEL = dict(
    NAME = 'SCOPSP',
    basename = 'SCOPSP',
    paras = dict(
        dim = lm_numb,
        in_channel = 3,
        h_channel = 32,
        mode = 'point',
        heatmap_std = 0.02,
        downsamplelast = True,
    )
)

# loss
metriclist_train = [
    {
        'name': 'PerceptualNewLoss',
        'paras': {
            'model_type': 'Vgg19',
            'content_layer': [0, 1, 2, 3, 4, 5],
            'loss_type': 'L2',
            'layer_weight': [1/0.25, 1/0.26, 1/1.4, 1/3.19, 1/2.54, 1/0.3],
            'item_map': [
                dict(x='img2', y='output.recovered_y', mask='img2_mask'),
                dict(x='img1', y='output.recovered_x', mask='img1_mask'),
            ],
            'refresh_iterval': 100,
            'return_single_loss': True,
        },
        'weight': 1.0
    },
    # {  
    #     'name': 'ArcFaceLoss',
    #     'paras': {
    #         'in_features': 256,
    #         'out_features': lm_numb,
    #         'item_map': [dict(feat='output.feats_x'), dict(feat='output.feats_y')],
    #         'm': 0.5
    #     },
    #     'weight': 1.0,
    # },
    # {
    #     'name': 'DeltaLoss',
    #     'paras': {
    #         'item_map': [
    #             dict(pred='output.hm_x_sm'),
    #             dict(pred='output.hm_y_sm'),
    #         ],
    #     },
    #     'weight': 1.0,
    # }
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

    regression_interval = 1,
    only_save_best_model = True,
    tb_plot_interval = 200,

    freeze_epoch = 60,

)


