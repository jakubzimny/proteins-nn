from parser import InputParser
from tensorflow import keras
from time import time
import numpy as np
   
if __name__== "__main__":
    start = time()

    input_type = 'B'
    token_increment = 10
    token_init = 10
    parser = InputParser(f'proteins/4r3o'+'HB',
                        input_type=input_type, token_increment=token_increment, token_init=token_init)
    data_ = parser.parse_input()
    data = [x for x in data_ if x !=[]]

    X, _ = parser.split_input_and_output(data)

    X = np.array(X, dtype=np.float32)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    data_load_time = time() - start
    start = time()

    model = keras.models.load_model(f'models/B_LSTM.h5')

    model_load_time = time() - start
    start = time()

    predicted = model.predict(X)
  
    inference_time = time() - start

    print(f'Data load time: {data_load_time}s')
    print(f'Model load time: {model_load_time}s')
    print(f'Inference time: {inference_time}s')
    print(f'Total time: {data_load_time + model_load_time + inference_time}s')