import youtokentome as yttm

def tokenize(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None)
    df_nonan = df.dropna()
    err = 0
    f = open('for_bpe_ctx_quest.txt', 'w')
    bpe_model_address = 'for_bpe_ctx_quest.txt'
    for que in df_nonan[0]:
        try:
            f.write(que + '\n')
        except:
            err += 1
    for que in df_nonan[1]:
        try:
            f.write(que + '\n')
        except:
            err += 1
    f.close()

    vocab_size = 16000
    model_path = 'bpe.model'
    yttm.BPE.train(data='for_bpe_ctx_quest.txt', vocab_size=vocab_size, model=model_path)
    tokenizer = yttm.BPE(model=model_path)

    tokenized_ctx = []
    tokenized_quest = []
    batch_size = 256
    for i_batch in tqdm(range(math.ceil(len(df_nonan[0]) / batch_size))):
        tokenized_ctx.extend(tokenizer.encode(list(df_nonan[0][i_batch*batch_size:(i_batch+1)*batch_size])))
 
    for i_batch in tqdm(range(math.ceil(len(df_nonan[1]) / batch_size))):
        tokenized_quest.extend(tokenizer.encode(list(df_nonan[1][i_batch*batch_size:(i_batch+1)*batch_size])))
        
    return tokenized_source, tokenized_target

