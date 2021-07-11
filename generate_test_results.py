from parser import InputParser
from model import model_A, model_lstm
from utils import split_train_test
from tensorflow import keras
import numpy as np
   
if __name__== "__main__":
    input_type = 'A'
    token_increment = 10
    token_init = 10
    train_test_ratio = 0.9
    parser = InputParser(f'data_extended/HA_All.inp',
                        input_type=input_type, token_increment=token_increment, token_init=token_init)
    data_ = parser.parse_input()
    data = [x for x in data_ if x !=[]]

    X, Y = parser.split_input_and_output(data)

    Y = np.array(Y, dtype=np.float32)
    X = np.array(X, dtype=np.float32)

    model = keras.models.load_model(f'models/B_FF.h5')

    predicted = model.predict(X)

    with open(f'results/B_FF.txt', 'w') as f:
            for i in range(0, len(X)):
                f.write(f'{Y[i][0]} {Y[i][1]} {Y[i][2]} {predicted[i][0]} {predicted[i][1]} {predicted[i][2]}\n')    
