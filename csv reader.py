import csv

with open('Bitcoin Historical Data.csv', newline='') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        print(row['Price'])
