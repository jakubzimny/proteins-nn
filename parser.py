from typing import List, Tuple

class AminoTokenizer:

    def __init__(self, token_increment: int):
        self._amino_dict = {}
        self._token_val = 0.0
        self._increment = token_increment

    def get_token(self, amino_name: str) -> float:
        val = self._amino_dict.get(amino_name)
        if val is None:
            val = self._token_val
            self._token_val += self._increment
            self._amino_dict[amino_name] = val
        return val


class InputParser:

    def __init__(self, file_name: str, input_type: str = 'A', token_increment: int = 1.0):
        self._file_name = file_name
        self._input_type = input_type
        self._tokenizer = AminoTokenizer(token_increment)

    def _create_data_row(self, data_list: List) -> List:
        row = []
        if self._input_type == 'A':
            if len(data_list) == 17: # single hydrogen
                for i in range(0, 8): # amino_token, theta, Ca x, Ca y, Ca z, Ha x, Ha y, Ha z   
                   row.append(float(data_list[i])) 
            else: # double hydrogen
                for i in range(0, 5): # amino_token theta, Ca x, Ca y, Ca z 
                    row.append(float(data_list[i])) 
                row.append((float(data_list[5]) + float(data_list[8])) / 2) # average Ha x
                row.append((float(data_list[6]) + float(data_list[9])) / 2) # average Ha y
                row.append((float(data_list[7]) + float(data_list[10])) / 2) # average Ha z
        elif self._input_type == 'B':
            for i in range(2, 9): # theta, Ca x, Ca y, Ca z, Qb x, Qb y, Qb z   
                row.append(float(data_list[i])) 
        elif self._input_type == 'N':
            for i in range(3, 14): # theta1, theta2, Ca x, Ca y, Ca z, Ca2 x, Ca2 y, Ca2 z, Hn x, Hn y, Hn z      
                row.append(float(data_list[i])) 
        else:
            print('Unknown input type, aborting')
            exit(0)
        return row

    def _parse_data_row(self, line: str, last_line: str = '') -> List:
        splitted_line = line.strip('\n').split(' ')
        data_list = []
        if last_line != '':
            amino_name = list(filter(('').__ne__, last_line.strip('\n').split(' ')))[1]
            data_list.append(str(self._tokenizer.get_token(amino_name)))
        for el in splitted_line:
            if el != '':
                data_list.append(el)
        data_row = self._create_data_row(data_list)
        return data_row
            

    def parse_input(self) -> List:
        with open(self._file_name) as input_file:
            counter = 0
            ignore_next = 0
            dataset = []
            last_line = ''
            for line in input_file:
                if counter != 0: # skip first line with number of pdbdata files
                    if 'pdbdata' in line: # ignoring file headers
                        ignore_next = 3
                    if ignore_next == 0:
                        if self._input_type == 'A' and counter % 2 == 0:
                            data_row = self._parse_data_row(line, last_line)
                            dataset.append(data_row)
                        elif self._input_type != 'A':
                            data_row = self._parse_data_row(line)
                            dataset.append(data_row)
                    else:
                        ignore_next -= 1
                        counter -= 1
                last_line = line
                counter += 1
            print(f'Parsed {counter} rows of data')
            return dataset
                    
    def split_input_and_output(self, dataset: List) -> Tuple:
        inp = []
        out = []
        for row in dataset:
            if self._input_type == 'A' or self._input_type == 'B':
                inp.append(row[:5])
                out.append(row[5:])
            elif self._input_type == 'N':
                inp.append(row[0:8])
                out.append(row[8:11])
            else:
                print('Unknown input type, aborting')
                exit(0)
        return (inp, out)