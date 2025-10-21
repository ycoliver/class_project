# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:02:26 2025
@author: Neal

Given that the shareholder information of a given stock ticker (such as 000001,
 000002 and 000003) are provided in :
    https://q.stock.sohu.com/cn/000001/ltgd.shtml
    https://q.stock.sohu.com/cn/000002/ltgd.shtml
    https://q.stock.sohu.com/cn/000003/ltgd.shtml
    ...

Please collect the shareholder information tables for the stocks listed in 
"selected_stocks," ensuring you include the following seven columns:


And you need to collect the tables of shareholder information for stocks in 
 "selected_stocks", with following 7 columns, 
    1. 'stock'-股票代码 / Stock code
    2. 'rank'-排名 / Shareholder rank
    3. 'org_name'-股东名称 / Shareholder name	
    4. 'shares'-持股数量(万股) / Number of shares held (in units of 10,000 shares)
    5. 'percentage'-持股比例	 / Shareholding percentage
    6. 'changes'-持股变化(万股) / Change in shares held (in units of 10,000 shares)
    7. 'nature'-股本性质 / Nature of equity
    
Then perform the analysis on the collected shareholder information to answer
 the questions.
   
Note:
    1. Be mindful of the default data types for each column, particularly 'rank' and 'percentage.'
    2. Remember that the 'shares' column is measured in increments of 10,000 shares.
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
chrome_header = {  "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
            "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding":"gzip, deflate, sdch",
            "Accept-Language":"zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4,zh-CN;q=0.2"
        }

data_file= './data/stock_shareholders.csv'
selected_stocks = ('601398','601857','601728','600276','601166','600887','601816','601328',
            '003816','300750','000333','300999','000651','300760','002415')


stock_share_prices ={'000333.CH': 73.41, '000651.CH': 40.11, '002415.CH': 30.92, '003816.CH': 3.68, 
            '300750.CH': 365.0, '300760.CH': 238.5, '300999.CH': 32.15, '600276.CH': 71.1, 
            '600887.CH': 27.73, '601166.CH': 20.28, '601328.CH': 6.89, '601398.CH': 7.28, 
            '601728.CH': 6.78, '601816.CH': 5.23, '601857.CH': 8.19}

print('There are', len(selected_stocks), 'stocks in selected_stocks')


def get_stock_info():

    base_url = 'https://q.stock.sohu.com/cn/{}/ltgd.shtml' 
    row_count = 0
    #create a list to store the crawled share-holdoing records
    results=[]

    #process stock one by one
    for stock in selected_stocks: 
        #prepare the request URL with desired parameters
        url = base_url.format(stock)     
        print("Now we are scraping stock",stock)
        #send http request with Chrome http header
        response = requests.get(url,headers = chrome_header)
        if response.status_code == 200:
            # response.encoding = #++insert your code here++  look for charset in html
            # response.encoding = response.apparent_encoding
            response.encoding = 'gbk' # gbk格式解析
             #parse the response text into a BeautifulSoup object
            root = BeautifulSoup(response.text,"html.parser") 
            # search the table storing the shareholder information
            # table = #++insert your code here++
            table = root.find('table', attrs={'class':'tableG'})
            # list all rows the table, i.e., tr tags
            # rows =  #++insert your code here++
            if table is None:
                print("No table found for stock", stock)
                continue
            rows = table.find_all('tr')[1:] # 跳过表头
            for row in rows: #iterate rows
                record=[stock+".CH",]# define a record with stock pre-filled and then store columns of the row/record
                # list all columns of the row , i.e., td tags
                # columns =  #++insert your code here++
                columns = row.find_all('td')
                for col in columns: #iterate colums
                    record.append(col.get_text().strip())
                if len(record) == 7:# if has valid columns, save the record to list results
                    #++insert your code here++
                    row_count+=1
                    results.append(record)
            time.sleep(1)
            
    sharehold_records_df = pd.DataFrame(columns=['stock', 'rank','org_name','shares','percentage','changes','nature'], data=results)
    print("\n", "="*20)
    print('Crawled and saved {} records of shareholder information of selected_stocks '.format(len(sharehold_records_df)) )
    sharehold_records_df.to_csv(data_file, index=False, encoding='utf-8')
    return sharehold_records_df, stock_share_prices

def get_target_hold(row):
    return row['nature'] == '境外可流通股' and row['stock'] in stock_share_prices

def Q2_1():
    sharehold_records_df = pd.read_csv('./data/stock_shareholders.csv')
    sharehold_records_df['is_target'] = sharehold_records_df.apply(get_target_hold,axis=1)
    target_df = sharehold_records_df[sharehold_records_df['is_target']]
    unique_org_names = target_df['org_name'].unique()
    print(f"The number of unique org_names with '境外可流通股' is {unique_org_names} with a total of {len(unique_org_names)} unique org_names")

def Q2_2():
    sharehold_records_df = pd.read_csv('./data/stock_shareholders.csv')
    sharehold_records_df['is_target'] = sharehold_records_df.apply(get_target_hold,axis=1)
    target_df = sharehold_records_df[sharehold_records_df['is_target']] 
    highest_df = target_df[target_df['rank'] == 1]
    # 统计org_name出现次数
    org_count = highest_df['org_name'].value_counts()
    frequent_org = org_count.index[0]
    print(f"The most frequent org with the highest number of shareholding records is {frequent_org}")
    return frequent_org

def Q2_3():
    total_share = 0
    top_stock = ''
    sharehold_records_df = pd.read_csv('./data/stock_shareholders.csv')
    group_df = sharehold_records_df.groupby('stock')['shares']
    for stock, group in group_df:
        gv = group.values
        gv = gv/gv.sum()
        if gv[:5].sum() > total_share:
            total_share = gv[:5].sum()
            top_stock = stock
    print(f"The stock with the highest proportion of shares held by its top 5 shareholders is {top_stock} with a proportion of {total_share:.2%}")
    return top_stock, total_share

def Q2_4():
    sharehold_records_df = pd.read_csv('./data/stock_shareholders.csv')
    sharehold_records_df['is_target'] = sharehold_records_df.apply(get_target_hold,axis=1)
    target_df = sharehold_records_df[sharehold_records_df['is_target']] 
    t = target_df
    sum_share_df = target_df.groupby('org_name')['shares'].sum().reset_index().sort_values(by='shares', ascending=False) # group以后求和然后倒序
    toporg = sum_share_df.iloc[0]['org_name']
    toporg_shares = sum_share_df.iloc[0]['shares']
    print(f'The org holding the largest total number of shares across all stocks is {toporg} with a total of {toporg_shares:.2f} shares')
    return toporg


if __name__ == "__main__":
    # get_stock_info()
    Q2_1()
    Q2_2()
    Q2_3()
    Q2_4()