import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


                                                        # This function fills nulls on Weight column
def fill_null_category(categ_li,dataframe):
    for category in categ_li:
        categ_ = dataframe[dataframe['Category'] == category]
        categ_mean = categ_['Weight'].mean(skipna=True)
        categ_ = dataframe.fillna({'Weight': categ_mean})
        return categ_


                                                        # This function replaces 'LF' and 'reg'
def clean_fat_content(dataframe):
    dataframe.replace(to_replace=['LF', 'reg'], value=['Low Fat', 'Regular'], inplace=True)
    return dataframe


                                                        # This function fills nulls on Store_Size column
def fill_null_store_size(dataframe):
    dataframe['Store_Size'].replace(to_replace=['High', 'Medium', 'Small'], value=[1, 2, 3], inplace=True)
    dataframe.fillna({'Store_Size': dataframe['Store_Size'].median()}, inplace=True)
    dataframe['Store_Size'].replace(to_replace=[1, 2, 3], value=['High', 'Medium', 'Small'], inplace=True)
    return dataframe


'''Bundas_Train Df Cleaning '''

bundas_train_df = pd.read_csv('C:/Users/user/Desktop/DataThon Challange/Bundas/bundas_train.csv')

bundas_train_df['Category'].value_counts()


                                # Creating list of categories
categ = ['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Canned',
         'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks', 'Others',
         'Starchy Foods', 'Breakfast','Seafood']


                                # Filled Train_df Category
categ_train_df = fill_null_category(categ, bundas_train_df)

                                # Cleaning FatContent
clean_fat_content(categ_train_df)

                                # Fill Store_Size
fill_null_store_size(categ_train_df)


'''Bundas_Test Df Cleaning '''

bundas_test_df = pd.read_csv('C:/Users/user/Desktop/DataThon Challange/Bundas/bundas_test.csv')


# Fill Null Category column of test df
categ_test_df = fill_null_category(categ,bundas_test_df)

# Clean FatContent Column
clean_fat_content(categ_test_df)

# Fill Store_Size
fill_null_store_size(categ_test_df)


categ_test_df['Item_Store_Sales'] = ''


'''MACHINE LEARNING MODEL'''


                                                                    # Initialize Our Training Values which are Numeric
merged_df = pd.concat([categ_train_df, categ_test_df])

"Code iS good For all things"

doub = lambda x:x*2

print(doub(3))









