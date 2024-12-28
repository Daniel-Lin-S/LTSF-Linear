import os
import pandas as pd
import os
from math import gcd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from abc import abstractmethod, ABC

warnings.filterwarnings('ignore')


class Base_Dataset(Dataset, ABC):
    """
    Base dataset class for prediction model training. \n
    Subclasses must define __read_data__.

    Notes
    -----
    - when mode is 'pred', the created y values
      will be taken from a window of length label_len + pred_len.
      Where label_len is overlapping with the window giving x-values.
      This is for transformer-based models to use next token prediction.
    """
    def __init__(self, root_path: str, flag: str, size,
                 features: str, data_path: str,
                 target: str, scale: bool,
                 timeenc: int, freq: str, train_only: bool,
                 hop_length: int, mode: str):
        """
        Parameters
        ----------
        root_path : str
            The root directory where the dataset is stored.
        flag : str, optional, default 'train'
            The type of dataset to load, can be 'train', 'val', or 'test'.
        size : list of int or None
            Specifies [seq_len, _, pred_len], where:
            - `seq_len` is the length of input sequences.
            - `label_len`: length of overlap for transformer
              based models.
            - `pred_len` is the length of the prediction.
        features : str
            The type of features in the dataset.
            Can be 'S', 'M', or 'MS'.
        data_path : str
            The path to the dataset file.
        target : str, optional, default 'OT'
            The target feature (column) to forecast.
        scale : bool, optional, default True
            Whether to scale the data using StandardScaler.
        timeenc : int
            Whether to include time-based features (like month, day, hour).
        freq : str
            Frequency of the data (e.g., 'h' for hourly data).
            Only valid when timeenc != 0.
        train_only : bool
            Whether the whole dataset should be treated as.
            training set
        hop_length : int
            interval between adjacent sliding windows to
            extract the dataset.
        mode : str, optional
            if mode is 'pred', items will be x-y pairs taken from
            adjacent sliding windows. If mode is 'recon',
            items are only x segments cut from the gcd of
            seq_len and pred_len. (unsupverised learning)
        """

        # initialise three lengths
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert mode in ['pred', 'recon']
        self.mode = mode
        if mode == 'recon':
            self.gcd_len = gcd(self.seq_len, self.pred_len)
            if self.gcd_len < 24:
                warnings.warn('length of greatest common divisor of seq_len and pred_len,'
                            f' {self.gcd_len}, is very short '
                            'consider changing seq_len and pred_len '
                            'for reconstructors to use more information', UserWarning)

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.hop_length = hop_length
        self.scaler = StandardScaler()

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self._validate_data()
    
    @abstractmethod
    def __read_data__(self):
        """
        Must assign self.data, self.data_stamp
        (time stamps of self.data)
        to be used by __getitem__
        """
        raise NotImplementedError

    def __getitem__(self, index: int):
        """
        Sliding window with hop length 1
        """
        if self.mode == 'pred':
            s_begin = index * self.hop_length
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data[s_begin:s_end]
            seq_y = self.data[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        elif self.mode == 'recon':
            start_idx = index * self.hop_length
            end_idx = start_idx + self.gcd_len

            seq_x = self.data[start_idx:end_idx]
            seq_x_mark = self.data_stamp[start_idx:end_idx]

            return seq_x, seq_x_mark
 
    def __len__(self):
        if self.mode == 'pred':
            return (len(self.data) - self.seq_len - self.pred_len
                    ) // self.hop_length + 1
        elif self.mode == 'recon':
            return (len(self.data) - self.gcd_len) // self.hop_length + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _construct_time_stamp(self, df_stamp: pd.DataFrame):
        """
        Turn `date` column into pre-defined time stamps.
        """
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data_stamp
    
    def _scale(self, border1s, border2s, df_data):
        """
        Fit and scale the data, using only the training
        data.
        """
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        return data

    def _validate_data(self):
        assert hasattr(self, 'data'), (
            "data is not assigned by read data method!"
        )
        assert hasattr(self, 'data_stamp'), (
            "data_stamp is not assigned by read data method!"
        )


class Dataset_ETT_hour(Base_Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False,
                 hop_length=1, mode='pred'):
        super().__init__(root_path, flag, size, features,
                         data_path, target, scale, timeenc, freq, train_only,
                         hop_length, mode)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0,
                    12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = self._scale(border1s, border2s, df_data)

        df_stamp = df_raw[['date']][border1:border2]
        data_stamp = self._construct_time_stamp(df_stamp)

        self.data = data[border1:border2]
        self.data_stamp = data_stamp


class Dataset_ETT_minute(Base_Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False,
                 hop_length=1, mode='pred'):
        super().__init__(root_path, flag, size, features,
                         data_path, target, scale, timeenc, freq, train_only,
                         hop_length, mode)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0,
                    12 * 30 * 24 * 4 - self.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = self._scale(border1s, border2s, df_data)

        df_stamp = df_raw[['date']][border1:border2]
        data_stamp = self._construct_time_stamp(df_stamp)

        self.data = data[border1:border2]
        self.data_stamp = data_stamp


class Dataset_Custom(Base_Dataset):
    """
    Class for getting x-y pairs for prediction model
    training.

    Notes
    -----
    Train-val-test ratio is fixed to 7:1:2. \n
    Data file being read must have the following columns
    ['date', ...(other features), target feature]
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False,
                 hop_length=1, mode='pred'):
        super().__init__(root_path, flag, size, features,
                         data_path, target, scale, timeenc, freq, train_only,
                         hop_length, mode)

    def __read_data__(self):
        """
        Reads and processes the raw data, applies scaling,
        and handles time-based features.
        """
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        border1s, border2s, border1, border2 = self._compute_borders(len(df_raw))

        # Process the data based on features
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        data = self._scale(border1s, border2s, df_data)

        df_stamp = df_raw[['date']][border1:border2]
        data_stamp = self._construct_time_stamp(df_stamp)

        self.data = data[border1:border2]
        self.data_stamp = data_stamp

    def _compute_borders(self, total_length):
        """
        Compute split borders for train-val-test splits.
        """
        num_train = int(total_length * (0.7 if not self.train_only else 1))
        num_test = int(total_length * 0.2)
        num_vali = total_length - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    total_length - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        return border1s, border2s, border1, border2


class Dataset_Pred(Dataset):
    """
    Data loadiing with no train-validation-test
    split.
    """
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False,
                 timeenc=0, freq='15min',
                 cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
