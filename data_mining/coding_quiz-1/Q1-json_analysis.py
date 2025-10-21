# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:47:45 2025

@author: Neal

Analyze the 10 provided JSON files (as below) and answer the questions in the answer book accordingly.
    1. nvda_1.json
    2. nvda_2.json
    3. nvda_3.json
    4. nvda_4.json
    5. nvda_5.json
    6. tsla_1.json
    7. tsla_2.json
    8. tsla_3.json
    9. tsla_4.json
   10. tsla_5.json

Hints:
    1. Skip/remove any duplicate posts that share the same "id".
    2. Assume a "followers_count" of 0 for any user object lacking a valid "followers_count."
"""

import json


def preanalysis():
    with open("./data/nvda_1.json",'r', encoding = "utf-8") as rf:
        nvdia_page_1 = json.load(rf)
        
    print('The 1st post object is: `nvdia_page_1["list"][0]`, and its "text" attribute is:\n {}'.format(
        nvdia_page_1['list'][0]['text']) )
        
    print('\n\n The user object of the 2nd post is: `nvdia_page_1["list"][1]["user"]`, and its "screen_name" attribute is: \n {}'.format(
        nvdia_page_1['list'][1]['user']['screen_name']) )

    #%% Hints for Q1-1
    print("\n", "="*20)
    print("Hints for Q1-1")
    total_post = 0
    unique_posts = set()
    for post in nvdia_page_1["list"]:
        total_post += 1
        unique_posts.add(post["id"])

    print(f"There are {len(unique_posts)} unique posts out of {total_post} total posts for `nvda_1.json`")

#++insert your code here++

def load_file(company,index):
    with open(f"./data/{company}_{index}.json",'r', encoding = "utf-8") as rf:
        page = json.load(rf)
    return page


def Q1_1():
    total_post = 0
    unique_posts = []
    unique_posts_ids = set()
    nvidia_post = 0
    tesla_post = 0
    for i in range(1,6):
        nvidia_page = load_file("nvda",i)
        for post in nvidia_page["list"]:
            total_post += 1
            if post['id'] not in unique_posts_ids:
                nvidia_post += 1
                unique_posts_ids.add(post['id'])
                unique_posts.append(post)
    for i in range(1,6):
        tsla_page = load_file("tsla",i)
        for post in tsla_page["list"]:
            total_post += 1
            if post['id'] not in unique_posts_ids:
                tesla_post += 1
                unique_posts_ids.add(post['id'])
                unique_posts.append(post)
    assert nvidia_post + tesla_post == len(unique_posts), "Post count mismatch"
    print(f"There are {len(unique_posts)} unique posts out of {total_post} total posts for all `json` files")
    print(f'Where {nvidia_post} posts are about NVIDIA and {tesla_post} posts are about Tesla.')
    return unique_posts

def Q1_2(unique_posts):
    target_post = 0
    for i in range(len(unique_posts)):
        source = unique_posts[i]['source']
        text = unique_posts[i]['text']
        if source == 'iPhone' and ('AI' in text or '人工智能' in text):
            target_post += 1 
    print(f'There are {target_post} posts that were posted from iPhone and contain both "AI" and "人工智能".')
    return target_post

def Q1_3(unique_posts):
    view_counts = []
    print(unique_posts[0].keys())
    print(unique_posts[0]['user'].keys())
    followers_counts = []
    # target_followers_count = 0
    for i in range(len(unique_posts)):

        if 'user' in unique_posts[i] and 'followers_count' in unique_posts[i]['user'] and unique_posts[i]['user']['followers_count'] > 100:
            followers_counts.append(unique_posts[i]['user']['followers_count'])
            view_counts.append(unique_posts[i]['view_count'])
    print('The average view_count of posts whose user has more than 100 followers is: {:.2f}'.format(sum(view_counts)/len(view_counts)))
    return sum(view_counts)/len(view_counts)

def Q1_4(unique_posts):
    view_counts = []
    followers_counts = []
    # target_followers_count = 0
    for i in range(len(unique_posts)):

        if 'user' in unique_posts[i] and 'followers_count' in unique_posts[i]['user'] and unique_posts[i]['user']['followers_count'] <= 100:
            followers_counts.append(unique_posts[i]['user']['followers_count'])
            view_counts.append(unique_posts[i]['view_count'])
    print('The average view_count of posts whose user has more than 100 followers is: {:.2f}'.format(sum(view_counts)/len(view_counts)))
    return sum(view_counts)/len(view_counts)

if __name__ == "__main__":
    preanalysis()
    unique_posts = Q1_1()
    Q1_2(unique_posts)
    Q1_3(unique_posts)
    Q1_4(unique_posts)
