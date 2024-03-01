if __name__ == '__main__':


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  
    config.gpu_options.allow_growth = True  
    set_session(tf.Session(config=config))


    t = time.time()
    maxlen = 20  
    config_path = '/home/codes/news_classify/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '/home/codes/news_classify/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '/home/codes/news_classify/chinese_L-12_H-768_A-12/vocab.txt'
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)


    tokenizer = OurTokenizer(token_dict)


    data_dir = '/home/codes/news_classify/comment_classify/'
    train_df = pd.read_csv(os.path.join(data_dir, 'union_train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))


    print(len(train_df), len(test_df))


    DATA_LIST = []
    for data_row in train_df.iloc[:].itertuples():
        DATA_LIST.append((data_row.content, data_row.label))
    DATA_LIST = np.array(DATA_LIST)


    DATA_LIST_TEST = []
    for data_row in test_df.iloc[:].itertuples():
        DATA_LIST_TEST.append((data_row.content, 0))
    DATA_LIST_TEST = np.array(DATA_LIST_TEST)


    n_cv = 5
    train_model_pred, test_model_pred = run_cv(n_cv, DATA_LIST, DATA_LIST_TEST)


    train_df['Prediction'] = train_model_pred
    test_df['Prediction'] = test_model_pred/n_cv


    train_df.to_csv(os.path.join(data_dir, 'train_union_submit2.csv'), index=False)


    test_df['ID'] = test_df.index
    test_df[['ID', 'Prediction']].to_csv(os.path.join(data_dir, 'submit2.csv'), index=False)


    auc = roc_auc_score(np.array(train_df['label']), np.array(train_df['Prediction']))
    print('auc', auc)


    print('time is ', time.time()-t)  # 2853s
