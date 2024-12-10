dataset_info = dict(
    dataset_name='Spine_dataset',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='k0', id=0, color=[51, 153, 255], type='', swap=''),
        1:
        dict(
            name='k1',
            id=1,
            color=[51, 153, 255],
            type='',
            swap=''),
        2:
        dict(
            name='k2',
            id=2,
            color=[51, 153, 255],
            type='',
            swap=''),
        3:
        dict(
            name='k3',
            id=3,
            color=[51, 153, 255],
            type='',
            swap=''),
        4:
        dict(
            name='k4',
            id=4,
            color=[51, 153, 255],
            type='',
            swap='')
        
    },
    # skeleton_info={
    #     0:
    #     dict(link=('k0', 'k1'), id=0, color=[255,0,0]),
    #     1:
    #     dict(link=('k1', 'k2'), id=1, color=[255,0,0]),
    #     2:
    #     dict(link=('k2', 'k3'), id=2, color=[255,0,0]),
    #     3:
    #     dict(link=('k3', 'k4'), id=3, color=[255,0,0]),

        
    # },
    joint_weights=[
        1., 1., 1., 1., 1.,
    ],
    sigmas=[]
)
# 获取关键点个数
# NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
# dataset_info['joint_weights'] = [1.0] * NUM_KEYPOINTS
# dataset_info['sigmas'] = [0.025] * NUM_KEYPOINTS