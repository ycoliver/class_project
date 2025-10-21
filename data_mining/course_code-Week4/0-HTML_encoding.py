# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:53:29 2025

@author: Neal
"""
import requests
from bs4 import BeautifulSoup


chrome_header = {  "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
            "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding":"gzip, deflate, sdch",
            "Accept-Language":"zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4,zh-CN;q=0.2"
        }
url_1 = "http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllMemordDetail.php?stockid=000001"
r1 = requests.get(url_1, headers = chrome_header)
# r1.encoding = "gb2312"
root = BeautifulSoup(r1.text,"html.parser") 
meta = root.find("meta")
print("\n==============\n" )
print("meta tag:", meta)
print("\n==============\n" )
print("meta tag with another encoding:", meta.encode("Latin-1"))


print("\n==============" )
print("==BS4 Gues==" )
print("==============\n" )

markup = b'''
 <html>
  <head>
   <meta content="text/html; charset=ISO-Latin-1" http-equiv="Content-type" />
  </head>
  <body>
   <p>Sacr\xe9 bleu!</p>
  </body>
 </html>
'''

soup = BeautifulSoup(markup, 'html.parser')
soup
print(soup)
