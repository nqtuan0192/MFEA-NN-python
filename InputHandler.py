import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing
class InputHandler:
    def ticTacToe(self, link):
        with open(link, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            #next(reader, None)

            input_tmp = []
            output_tmp = []
            data_list = list(reader)
            print(data_list)
            np.random.shuffle(data_list)
            for row in data_list:
                tmp = []
                for i in range(0, len(row) - 1):
                    if(row[i] == 'x'):
                        tmp.append(1)
                    elif(row[i] == 'o'):
                        tmp.append(0)
                    else:
                        tmp.append(-1)
                output_tmp.append(tmp)
                if(row[-1] == 'positive'):
                    input_tmp.append(1)
                else:
                    input_tmp.append(-1)

            input_arr = np.array(input_tmp)
            output_arr = np.array(output_tmp)

            print(input_arr)
            print(output_arr)
            return input_arr, output_arr
        pass
    def ionosphere(self, link):
        with open(link, "r") as f:
            reader = csv.reader(f, delimiter=',')

            in_tmp = []
            out_tmp = []

            data_list = list(reader)
            for row in data_list:
                tmp = row[:-1:]
                in_tmp.append(tmp)
                out_tmp.append(row[-1])
            in_arr = np.array(in_tmp)
            out_arr = np.array(out_tmp)
            print(in_arr)
            print(out_arr)
            return in_arr, out_arr 
    def creditScreening(self, link):
        df = pd.read_csv(link, header=None, index_col=None)

        min_max = preprocessing.MinMaxScaler()
        labelEncoder = preprocessing.LabelEncoder()
        delete_idx = []
        try:
            for ridx in df.index.values:
                #print(df.iloc[ridx])
                for field in df.iloc[ridx]:
                    #print(field)
                    if field == '?':
                        delete_idx.append(ridx)
                        break
            for i in reversed(delete_idx):
                df.drop(i, inplace=True)
            pass
            print(df)
        except ValueError as e:
            print(ridx)
            pass
        # except IndexError as e_i:
        #     print(ridx)
        #     pass
        df = df.convert_objects(convert_numeric=True)
        for col in df.columns.values:
            #print(df[col].dtypes)
            if df[col].dtypes == 'object':
                #print(col)
                data = df[col].append(df[col])
                labelEncoder.fit(data.values)
                df[col] = labelEncoder.transform(df[col])
        
    
        data_minmax = min_max.fit_transform(df)
        dlen = len(data_minmax[0])
        out_arr = np.array(data_minmax[::1, dlen - 1])
        in_arr = np.array([data_minmax[::1, -1]])
        return in_arr, out_arr
        pass
        
                    


