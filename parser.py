from typing import List, Tuple
from math import sqrt


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
        self._file_counters = []  # number of records of each pbdata file
        self._neighbouring_records = []  # list of records with hydrogen less than 8 angstroms away for each record 
        self._record_counter = 0
        self._dataset = []

    def _create_data_row(self, data_list: List) -> List:
        row = []
        if self._input_type == 'A':
            if len(data_list) == 17: # single hydrogen
                # data_list expected: amino_token, theta, Ca x, Ca y, Ca z, Ha x, Ha y, Ha z, Vx[xyz], Vy[xyz], Vz[xyz]   
                row.append(float(data_list[0]))
                #row.append(float(data_list[1]) * 180.0 / 3.14159)
                row.append(float(data_list[1]))
                for i in range(2, 5):
                    #row.append(float(data_list[i]) + float(data_list[i+6]) + float(data_list[i+9]) + float(data_list[i+12]))
                    row.append(float(data_list[i]))
                for i in range(5, 8):
                    row.append(float(data_list[i]))
            else: # double hydrogen
                row.append(float(data_list[0]))
                #row.append(float(data_list[1]) * 180.0 / 3.14159)
                row.append(float(data_list[1]))
                for i in range(2, 5):
                    #row.append(float(data_list[i]) + float(data_list[i+9]) + float(data_list[i+12]) + float(data_list[i+15]))
                    row.append(float(data_list[i]))
                for i in range(5, 8):
                    row.append((float(data_list[i]) + float(data_list[i+3])) / 2) # average Ha
        elif self._input_type == 'B':
            row.append(self._tokenizer.get_token(data_list[1]))
            #row.append(float(data_list[2]) * 180.0 / 3.14159) # theta
            row.append(float(data_list[2]))
            for i in range(3, 6):
                # Ca x, Ca y, Ca z
                #row.append(float(data_list[i]) + float(data_list[i+6]) + float(data_list[i+9]) + float(data_list[i+12]))
                row.append(float(data_list[i]))
            for i in range(6, 9): # Qb x, Qb y, Qb z  
                row.append(float(data_list[i])) 
        elif self._input_type == 'N':
            for i in range(3, 14): # theta1, theta2, Ca x, Ca y, Ca z, Ca2 x, Ca2 y, Ca2 z, Hn x, Hn y, Hn z      
                row.append(float(data_list[i])) 
        else:
            print('Unknown input type, aborting')
            exit(0)
        self._record_counter += 1
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
        #TODO refactor
        with open(self._file_name) as input_file:
            counter = 0
            ignore_next = 0   
            last_line = ''
            last_file_count = self._record_counter
            for line in input_file:
                if counter != 0: # skip first line with number of pdbdata files
                    if '.pdb' in line: # ignoring file headers             
                        ignore_next = 3
                        if counter != 1:
                            self._file_counters.append(self._record_counter - last_file_count)
                            last_file_count = self._record_counter
                    if ignore_next == 0:
                        if self._input_type == 'A' and counter % 2 == 0:
                            data_row = self._parse_data_row(line, last_line)
                            self._dataset.append(data_row)
                        elif self._input_type != 'A':
                            data_row = self._parse_data_row(line)
                            self._dataset.append(data_row)
                    else:
                        ignore_next -= 1
                        counter -= 1
                last_line = line
                counter += 1
            self._file_counters.append(self._record_counter - last_file_count)  # account for last file
            print(f'Parsed {counter} rows of data')
            return self._dataset
                    
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

    def get_file_index_and_offset(self, record_index: int) -> Tuple:
        counter_sum = 0
        for index, counter in enumerate(self._file_counters):
            counter_sum += counter
            if record_index <= counter_sum:
                return (index, counter_sum - counter)

    def _populate_neighbours_list(self):
        for record_idx, record in enumerate(self._dataset):
            file_idx, file_offset = self.get_file_index_and_offset(record_idx)
            neighbours = [] 
            if (self._input_type == 'N'):
                x1 = record[8]
                y1 = record[9]
                z1 = record[10]
            else:
                x1 = record[5]
                y1 = record[6]
                z1 = record[7]
            for i in range(file_offset, file_offset + self._file_counters[file_idx]):
                if i == record_idx:
                    continue
                if (self._input_type == 'N'):
                    x2 = self._dataset[i][8]
                    y2 = self._dataset[i][9]
                    z2 = self._dataset[i][10]
                else:
                    x2 = self._dataset[i][5]
                    y2 = self._dataset[i][6]
                    z2 = self._dataset[i][7]
                distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                if distance < 8:
                    neighbours.append(i)
            self._neighbouring_records.append(neighbours)
        
    def get_neighbouring_records_data(self) -> List:
        self._populate_neighbours_list()
        return self._neighbouring_records
