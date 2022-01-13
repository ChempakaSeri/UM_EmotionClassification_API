from init_v2 import app,load_model,os,pickle,keras_seq,request,jsonify,pickle,np,textstat,flask,warnings,pred_models

def main():
    # ## Load the Keras-Tensorflow models into a dictionary
    # global pred_models 
    
    # pred_models={'word2seq_cnn' : load_model('./Models/word2seq_cnn.hdf5'),
    #              'word2vec_cnn' : load_model('./Models/word2vec_cnn.hdf5'),
    #              'word2seq_cnn_birnn_bilstm' : load_model('./Models/word2seq_cnn_birnn_bilstm.hdf5'),
    #              'word2vec_cnn_birnn_bilstm' : load_model('./Models/word2vec_cnn_birnn_bilstm.hdf5')}
    
    ## Make prediction function
    # for model in [model[:-5]for model in os.listdir('./Models')[1:]]:
    #     pred_models[model]._make_predict_function()
    
    # ## Loading the Keras Tokenizer sequence file
    # global tokenizer
    # with open('./pickle/tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)
        
    app.run(host='localhost', port=5000)

if __name__ == '__main__':
    main()


