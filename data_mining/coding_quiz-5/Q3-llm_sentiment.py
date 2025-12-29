# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:15:28 2025

@author: Neal
Please use ZhipuAI model-"glm-4-flash" and appropriate prompt to analyze the 
sentiment of news titles following steps as below.
    1)	Install the ZhipuAI SDK package as "pip install zhipuai" if necessary.
    2)	Register your ZhipuAI account and get your API token, and fill in it 
        in the code indicated as "++insert your API token ++".
    3)	Complete the definition of function "analyze_sentiment()" with 
        appropriate prompt.
    4)	Fine-tune your prompt and code in function "analyze_sentiment()" based 
        on its performance (in terms of accuracy) on (all or selected) labelled
        news sentiment samples in the file "benchmark_news.xlsx" 
    5)	If the performance is satisfactory (e.g., accuracy >0.9), then 
        apply the function "analyze_sentiment()" to analyze the 50 "news_title" 
        in "test_news.xlsx", and saved the returned "score","reason" as two 
        new columns in the name of "pred_sentiment" and "sentiment_reason", respectively.
    6)	Save the results in a new Excel file named as 
        "test_news_with_predictions.xlsx".This file is expected to 
        have 50 rows for 50 "news_title" with 4 columns as below
        ('news_id', 'news_title', 'pred_sentiment', 'sentiment_reason'). 

Note:
    1. Improve your prompt as in lecture notes of Week-12, especially the 
        slide-"Prompt Engineering in General",
    2. Try to restrict the output format (such as json) in prompt, such that  
       you could parse it reliably and easily
    3. Use "try ... except ..." clause to deal with unexpected situations
    4. Refer to the "./data/[Example]test_news_with_predictions.xlsx" as 
       an example of file format(/columns) to be submitted 
"""

import json
import pandas as pd
from zhipuai import ZhipuAI
from sklearn.metrics import accuracy_score
import os
expected_score = {1, -1}


def analyze_sentiment(glm_model, text): 
    """
    Analyze the sentiment of given `text` with glm_model ("glm-4-flash")
    Parameters
    ----------
    glm_model : zhipuai._client.ZhipuAI
        The authorized client to use ZhipuAI
    text : str
        The text to be analyzed

    Returns
    -------
    score : int
        The sentiment score, should be either 1 or -1
    reason : str
        The reason for the provided sentiment score

    """
    #++insert your code below ++ to compute `score` and `reason`
    # for `text` based on `glm_model`
    
    # Define the prompt for sentiment analysis
    prompt = f"""你是一个专业的新闻情感分析专家。请分析以下新闻标题的情感倾向。

任务说明：
- 分析新闻标题传达的整体情感倾向
- 正面情感(positive)：表示好消息、积极发展、成功、增长、利好等
- 负面情感(negative)：表示坏消息、问题、失败、下降、风险、损失等

新闻标题："{text}"

请严格按照以下JSON格式输出，不要输出任何其他内容：
{{"score": 1或-1, "reason": "简短解释原因"}}

其中：
- score为1表示正面情感
- score为-1表示负面情感

请直接输出JSON，不要添加任何markdown格式或其他文字。"""

    try:
        # Call the ZhipuAI API
        response = glm_model.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是一个专业的新闻情感分析助手，只输出JSON格式的结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent output
        )
        
        # Extract the response content
        result_text = response.choices[0].message.content.strip()
        
        # Clean the response (remove potential markdown formatting)
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON response
        result = json.loads(result_text)
        score = int(result["score"])
        reason = result["reason"]
        
        # Validate score
        if score not in expected_score:
            # Default to -1 if invalid score
            score = -1 if "负" in reason or "下" in reason or "风险" in reason else 1
            
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract sentiment from text
        print(f"JSON解析错误: {e}, 原始响应: {result_text}")
        if "1" in result_text and "-1" not in result_text:
            score = 1
            reason = "解析错误，根据响应推断为正面"
        else:
            score = -1
            reason = "解析错误，根据响应推断为负面"
            
    except Exception as e:
        # Handle any other exceptions
        print(f"API调用错误: {e}")
        score = -1
        reason = f"分析过程出现错误: {str(e)}"
    
    assert score in expected_score
    return score, reason



if __name__ == "__main__":
    
    #++insert your API token ++ of  ZhipuAI after registration
    client = ZhipuAI(api_key="0d2b5ac60e244ec6bee5c104e39ccaf7.PMx1VbtJwx9MckVK") 
    
    # %% Evaluate and improve the performance of your solution on "benchmark_news.xlsx"
    benchmark_file = os.path.join('./data', "benchmark_news.xlsx")
    df_benchmark = pd.read_excel(benchmark_file)
    print(f"\n========Processing {benchmark_file}==========")
    pred_sentiments = []
    true_sentiments = []
    count = 0
    for row in df_benchmark.itertuples(index=False):
        count+=1
        print(f"Process {count} rows with {row.news_title}")
        score, reason = analyze_sentiment(client, row.news_title)
        pred_sentiments.append(score)
        true_sentiments.append(row.news_sentiment)
    
    accuracy = accuracy_score(true_sentiments, pred_sentiments)
    print(f"\nAccuracy_score among {len(true_sentiments)} news in {benchmark_file} is", 
          round(accuracy, 4))

    # %% Predict the `pred_sentiment` and `sentiment_reason` for `news_title` 
    # in "test_news.xlsx" and save to Excel (to be submit)
    test_file = os.path.join('./data', "test_news.xlsx")
    df_test = pd.read_excel(test_file)
    print(f"\n========Processing {test_file}==========")
    results = []
    count = 0
    for row in df_test.itertuples(index=False):
        count+=1
        # 注意：提交时需要处理全部50条，去掉下面的break限制
        # if count > 5:
        #     break
        print(f"Process {count} rows with {row.news_title}")
        score, reason = analyze_sentiment(client, row.news_title)
        results.append((row.news_id, row.news_title, score, reason))
    
    df_results = pd.DataFrame(results,
                          columns=['news_id', 'news_title', 'pred_sentiment', 'sentiment_reason'])
    out_file = os.path.join('./data', "test_news_with_predictions.xlsx")
    df_results.to_excel(out_file, index=False)
    print(f'\nSave {len(results)} results to {out_file}')
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Q3-1: Accuracy score on benchmark_news.xlsx: {round(accuracy, 4)}")
    print(f"Q3-2: Results saved to: {out_file}")
    print(f"      Total predictions: {len(results)} news titles")
