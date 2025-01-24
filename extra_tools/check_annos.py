import pickle
import sys

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_dict_one_by_one(data):
    for d in data:
        breakpoint()
        if isinstance(d, dict):
            for key, value in d.items():
                print(f"{key}: {value}\n")
        else:
            print("The loaded data is not a dictionary.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_annos.py /path/to/your/pickle_file.pkl")
        sys.exit(1)

    file_path = sys.argv[1]  # 여기에 pkl 파일 경로를 입력하세요
    data = load_pickle(file_path)
    print_dict_one_by_one(data)