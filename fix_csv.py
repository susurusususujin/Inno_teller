import csv

with open('neurips_data.csv', 'r') as f:
    lines = f.readlines()

with open('neurips_data_clean.csv', 'w') as f:
    for line in lines:
        if line.count('"') % 2 != 0:
            # If quotes are open, just close them at the end of the line
            line = line.rstrip('\n') + '"\n'
        f.write(line)
