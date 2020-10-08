import csv

txt_file = r"./average_list.txt"
csv_file = r"./average_list.csv"

in_text = csv.reader(open(txt_file, "r"))
out_csv = csv.writer(open(csv_file, "w"))
out_csv.writerows(in_text)