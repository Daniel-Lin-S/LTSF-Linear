from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    """
    Create the dataset and data loader

    Parameters
    ----------
    args : object
        specifies basic settings including:
        - data (str): name of the dataset used
          for ETT data, this must be specified as
          one of ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'];
          for other data sets, use 'custom'
        - embed (str): time stamp representation strategy
          'timeF' - stamped by month, day,
          weekday, hour. Otherwise, use the frequency
          specified by args.freq
        - freq (str): the specific time stamping frequency
          used when embed is not 'timeF'.
          See utils.timefeatures for details
        - root_path (str): path to the folder storing the data file
        - data_path (str): name of the data file, e.g. 'weather.csv'
        - batch_size (int): number of samples in each batch
        - features (str): strategy of handling channels,
          must be one of ['M', 'S', 'MS']
          M - multivariate;
          S - univariate;
          MS - multivariate dataset, but only predict
          the LAST variable.
        - target (str): column name of the feature being
          predicted when feature is S or MS.
        - num_workers (int): number of parallel workers
          for data loading

    flag: str
        the strategy of getting data.
        Must be one of 'test', 'val', 'train', 'pred'
    
    Return
    ------
    Dataset, DataLoader

    Notes
    -----
    Each x (data) would have length seq_len,
    and each y (label) would have length label_len + pred_len
    with the first label_len entries overlapping with x.
    """

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:  # training/validation
        shuffle_flag = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )

    n_iters = len(data_set) // batch_size
    print(flag, f'{len(data_set)} samples', f'{n_iters} batches')

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False)  # avoid unfair comparison caused by drop_last
    return data_set, data_loader
