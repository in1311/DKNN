from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataSet():
    ##### read and load dataset #####
    def __init__(self, data):
        super(DataSet, self).__init__()
        self.data = data
        self.train_data, self.val_data, self.test_data = self.load_dataset(self.data)
        self.train_num = len(self.train_data)

    def scaler_data(self):
        ##### scale dataset #####
        self.scaler_coods = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features = StandardScaler()
        self.scaler_label = StandardScaler()
        data_train_scaler = self.train_data.copy()
        data_val_scaler = self.val_data.copy()
        data_test_scaler = self.test_data.copy()

        data_train_scaler.index = data_train_scaler.values[:,0]
        data_val_scaler.index = data_val_scaler.values[:,0]
        data_train_scaler.iloc[:,0] = range(0, len(self.train_data))
        data_val_scaler.iloc[:,0] = range(0, len(self.val_data))

        ##### scale coods #####
        data_train_scaler.iloc[:,1:3] = self.scaler_coods.fit_transform(self.train_data.values[:,1:3])
        data_val_scaler.iloc[:, 1:3] = self.scaler_coods.transform(self.val_data.values[:, 1:3])
        if data_test_scaler.empty is not True:
            data_test_scaler.iloc[:, 1:3] = self.scaler_coods.transform(self.test_data.values[:, 1:3])
            data_test_scaler.index = data_test_scaler.values[:,0]
            data_test_scaler.iloc[:,0] = range(0, len(self.test_data))

        ##### scale auxiliary variables #####
        data_train_scaler.iloc[:,3:-1] = self.scaler_features.fit_transform(self.train_data.values[:,3:-1])
        data_val_scaler.iloc[:,3:-1] = self.scaler_features.transform(self.val_data.values[:,3:-1])
        if data_test_scaler.empty is not True:
            data_test_scaler.iloc[:,3:-1] = self.scaler_features.transform(self.test_data.values[:,3:-1])

        ##### scale target variable #####
        data_train_scaler.iloc[:,-1] = self.scaler_label.fit_transform(self.train_data.values[:,-1].reshape(-1,1))
        data_val_scaler.iloc[:,-1] = self.scaler_label.transform(self.val_data.values[:,-1].reshape(-1,1))
        if data_test_scaler.empty is not True:
            data_test_scaler.iloc[:,-1] = self.scaler_label.transform(self.test_data.values[:,-1].reshape(-1,1))

        return {'train':data_train_scaler, 'val':data_val_scaler, 'test':data_test_scaler}

    def load_dataset(self, all_data):
        ##### load dataset #####
        if 'z_trend' in all_data.columns:
            all_data = all_data.drop('z_trend',axis=1, inplace=False)
        train_data = all_data[all_data['dataset'] == 'train']
        val_data = all_data[all_data['dataset'] == 'val']
        test_data = all_data[all_data['dataset'] == 'test']
        train_data = train_data.drop('dataset',axis=1, inplace=False)
        val_data = val_data.drop('dataset',axis=1, inplace=False)
        test_data = test_data.drop('dataset',axis=1, inplace=False)

        return train_data, val_data, test_data

    def get_data(self):
        return self.train_data, self.val_data, self.test_data