负责向量相似度检索

流程:
    
    1. 构建索引库(query-answer一一对应), query已经从words转为vector, word 2 vector使用bert编码, 
       或者使用腾讯的词向量(通过语言模型训练的---因果关系语言模型, 例如Glove)分词之后均值转换
    2. 向量召回(向量检索) --- 获取相似向量的索引号, 之后去数据库查询对应answer

参考资料:
    
    1. https://github.com/jd-aig/nlp_baai/tree/master/jddc2020_baseline/mddr(客服多轮对话的检索模型)
    