import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import svr_model, rf_model, lr_model, tune_model

PATH = 'data/DatArticle_orig.xls'

# Define functions
def split_X_y(df):
    """Split the dataframe to X and y"""
    X = df[[('Temperature', 'norm'), ('mu', 'norm')]]
    y = df[[('qp', 'norm')]]
    return X, y

def main():                                            
    df = pd.read_excel(PATH, header=[0, 1])
    print(f'Dataset shape: {df.shape}')

    # Use 1 run for testing and all the other for training
    run_ids = df[('Run','id')].unique()
    
    # Create res_df
    res_df = pd.DataFrame()

    for run in run_ids:
        print(f'Run: {run}')

        # initialize results dataframe
        results = pd.DataFrame(columns=['run', 'sample_id', 'y_true', 'y_pred_svr', 
                                        'y_pred_rf', 'y_pred_lr', 'mse_svr', 'mse_rf', 'mse_lr'])

        test = df[df[('Run','id')] == run]
        train  = df[df[('Run','id')] != run]

        print(f'Training set size: {train.shape}')
        print(f'Testing set size:  {test.shape}')

        # split train and test to X and y
        X_train, y_train = split_X_y(train)
        X_test, y_test = split_X_y(test)

        # train models
        svr = svr_model(X_train, y_train)
        rf = rf_model(X_train, y_train)
        lr = lr_model(X_train, y_train)

        # predictions
        y_pred_svr = svr.predict(X_test)
        y_pred_rf = rf.predict(X_test)
        y_pred_lr = lr.predict(X_test)

        # mse calculations
        mse_svr = mean_squared_error(y_test, y_pred_svr)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        mse_lr = mean_squared_error(y_test, y_pred_lr)

        # update results df
        results['run'] = [run] * X_test.shape[0]
        results['sample_id'] = y_test.index
        results['y_true'] = y_test.values
        results['y_pred_svr'] = y_pred_svr
        results['y_pred_rf'] = y_pred_rf
        results['y_pred_lr'] = y_pred_lr
        results['mse_svr'] = mse_svr
        results['mse_rf'] = mse_rf
        results['mse_lr'] = mse_lr
        
        # concat results to res_df
        res_df = pd.concat([res_df, results], ignore_index=True)

    # save results
    res_df.to_csv('data/results_allruns.csv', index=False)

if __name__ == '__main__':
    main()
