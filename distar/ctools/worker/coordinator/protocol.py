import torch
import io


BYTES_LENGTH = 5
DATA_LENGTH = 4
DICT_KEY_LENGTH = 1
DICT_B = b'\x01'
DICT_E = b'\x10'
LIST_B = b'\x02'
LIST_E = b'\x20'
LIST_ITEM_B = b'\x03'
KEY_B = b'\x04'
DATA_B = b'\x05'


def encode(data, bytes=None, list_bytes=None, entry=True):
    if entry:
        bytes = io.BytesIO()
        list_bytes = []
    if isinstance(data, list):
        bytes.write(LIST_B)
        for d in data:
            bytes.write(LIST_ITEM_B)
            encode(d, bytes=bytes, list_bytes=list_bytes, entry=False)
        bytes.write(LIST_E)
    elif isinstance(data, dict):
        bytes.write(DICT_B)
        for k, v in data.items():
            bytes.write(KEY_B)
            k_bytes = k.encode(encoding='utf8')
            k_bytes_len = len(k_bytes)
            bytes.write(k_bytes_len.to_bytes(length=DICT_KEY_LENGTH, byteorder='big', signed=False))
            bytes.write(k_bytes)
            encode(v, bytes=bytes, list_bytes=list_bytes, entry=False)
        bytes.write(DICT_E)
    elif isinstance(data, torch.Tensor):
        bytes.write(DATA_B)
        b = io.BytesIO()
        torch.save(data, b)
        data_bytes = b.getvalue()
        data_length = len(data_bytes)
        bytes.write(data_length.to_bytes(length=DATA_LENGTH, byteorder='big', signed=False))
        list_bytes.append(bytes.getvalue())
        list_bytes.append(data_bytes)
        bytes.truncate(0)
        bytes.seek(0)
    if entry:
        list_bytes.append(bytes.getvalue())
        bytes_length = sum([len(b) for b in list_bytes])
        list_bytes.insert(0, bytes_length.to_bytes(length=BYTES_LENGTH, byteorder='big', signed=False))
        return list_bytes


def decode(bytes):
    data_type = bytes.read(1)
    if data_type == LIST_B:
        list_data = []
        while True:
            list_item_type = bytes.read(1)
            if list_item_type == LIST_E:
                return list_data
            elif list_item_type == LIST_ITEM_B:
                data = decode(bytes)
                list_data.append(data)
            else:
                raise ValueError(f'invalid data type: {list_item_type} when reading list data')
    elif data_type == DICT_B:
        dict_data = {}
        while True:
            dict_item_type = bytes.read(1)
            if dict_item_type == DICT_E:
                return dict_data
            elif dict_item_type == KEY_B:
                key_length = int.from_bytes(bytes.read(DICT_KEY_LENGTH), byteorder='big', signed=False)
                key = bytes.read(key_length).decode('utf8')
                dict_data[key] = decode(bytes)
            else:
                raise ValueError(f'invalid data type: {dict_item_type} when reading dict data')
    elif data_type == DATA_B:
        data_length = int.from_bytes(bytes.read(DATA_LENGTH), byteorder='big', signed=False)
        data_bytes = bytes.read(data_length)
        data_io = io.BytesIO(data_bytes)
        data = torch.load(data_io)
        return data

