from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataSet():
    ##### read and load dataset #####
    def __init__(self, data):
        super(DataSet, self).__init__()
        self.data = data
        self.train_data, self.test_data = self.load_dataset(self.data)
        self.train_num = len(self.train_data)

    def scaler_data(self):
        
        ##### scale dataset #####
        self.scaler_coods = MinMaxScaler(feature_range=(0, 1))  # MinMaxScaler for coodianates
        self.scaler_features = StandardScaler()  # StandardScaler for auxiliary variables
        self.scaler_label = StandardScaler()  # StandardScaler for target variable

        data_train_scaler = self.train_data.copy()
        data_test_scaler = self.test_data.copy()

        # reset index
        data_train_scaler.index = data_train_scaler.values[:,0]
        data_test_scaler.index = data_test_scaler.values[:,0]
        data_train_scaler.iloc[:,0] = range(0, len(self.train_data))
        data_test_scaler.iloc[:,0] = range(0, len(self.test_data))

        ##### scale coods #####
        data_train_scaler.iloc[:,1:3] = self.scaler_coods.fit_transform(self.train_data.values[:,1:3])  # fit and transform
        data_test_scaler.iloc[:, 1:3] = self.scaler_coods.transform(self.test_data.values[:, 1:3])  # transform

        ##### scale auxiliary variables #####
        data_train_scaler.iloc[:,3:-1] = self.scaler_features.fit_transform(self.train_data.values[:,3:-1])  # fit and transform
        data_test_scaler.iloc[:,3:-1] = self.scaler_features.transform(self.test_data.values[:,3:-1])  # transform

        ##### scale target variable #####
        data_train_scaler.iloc[:,-1] = self.scaler_label.fit_transform(self.train_data.values[:,-1].reshape(-1,1))  # fit and transform
        data_test_scaler.iloc[:,-1] = self.scaler_label.transform(self.test_data.values[:,-1].reshape(-1,1))  # transform

        return {'train':data_train_scaler, 'test':data_test_scaler}

    def load_dataset(self, all_data):
        ##### load dataset #####
        if 'trend' in all_data.columns:
            all_data = all_data.drop('trend',axis=1, inplace=False)
        train_data = all_data[all_data['dataset'] == 'train']
        test_data = all_data[all_data['dataset'] == 'test']
        train_data = train_data.drop('dataset',axis=1, inplace=False)
        test_data = test_data.drop('dataset',axis=1, inplace=False)

        return train_data, test_data

    def get_data(self):
        return self.train_data, self.test_data