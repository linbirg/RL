import csv

def load_cvs(file_path):
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        return list(csv_reader)