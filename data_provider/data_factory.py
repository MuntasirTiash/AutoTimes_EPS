from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_M4, Dataset_Solar, Dataset_TSF, Dataset_TSF_ICL
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_provider.panel_cov_loader import Dataset_PanelCov

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
    'tsf': Dataset_TSF,
    'tsf_icl': Dataset_TSF_ICL,
    'panel_cov': Dataset_PanelCov,
}


# def data_provider(args, flag):
#     Data = data_dict[args.data]

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = args.batch_size 
#     elif flag == 'val':
#         shuffle_flag = args.val_set_shuffle
#         drop_last = False
#         batch_size = args.batch_size 
#     else:
#         shuffle_flag = True
#         drop_last = args.drop_last
#         batch_size = args.batch_size

#     if flag in ['train', 'val']:
#         data_set = Data(
#             root_path=args.root_path,
#             data_path=args.data_path,
#             flag=flag,
#             size=[args.seq_len, args.label_len, args.token_len],
#             seasonal_patterns=args.seasonal_patterns,
#             drop_short=args.drop_short,
#         )
#     else:
#         data_set = Data(
#             root_path=args.root_path,
#             data_path=args.data_path,
#             flag=flag,
#             size=[args.test_seq_len, args.test_label_len, args.test_pred_len],
#             seasonal_patterns=args.seasonal_patterns,
#             drop_short=args.drop_short,
#         )
#     if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
#         print(flag, len(data_set))
#     if args.use_multi_gpu:
#         train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
#         data_loader = DataLoader(data_set, 
#             batch_size=batch_size,
#             sampler=train_datasampler,
#             num_workers=args.num_workers,
#             persistent_workers=True,
#             pin_memory=True,
#             drop_last=drop_last,
#             )
#     else:
#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last)
#     return data_set, data_loader

def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag in ['train', 'val']:
        kw = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.token_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
        )
    else:
        kw = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.test_label_len, args.test_pred_len],
            seasonal_patterns=args.seasonal_patterns,
            drop_short=args.drop_short,
        )

    # Pass panel-specific kwargs
    if args.data == 'panel_cov':
        kw.update(
            id_col=args.panel_id_col,
            time_col=args.panel_time_col,
            y_col=args.panel_y_col,
            cov_cols=[c.strip() for c in args.panel_cov_cols.split(',')] if args.panel_cov_cols else None,
        )

    data_set = Data(**kw)
    print(flag, len(data_set))

    # Build the loader expected by callers
    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')
    num_workers = getattr(args, 'num_workers', 0)

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=getattr(args, 'pin_memory', False),
    )

    return data_set, data_loader