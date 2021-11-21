import os
import csv

def create_csv_rows(dataframe_dir, labels):
	rows = []
	for root_dir, dirs, files in os.walk(dataframe_dir):
		if root_dir == dataframe_dir:
			continue
			
		for class_root_dir, class_dirs, class_files in os.walk(root_dir):
			for file_ in class_files:
				rows.append([file_, labels.index(root_dir.split('/')[-1]), class_root_dir[class_root_dir.find('/')+1:] + '/'])
	return rows
            
def write_csv_file(filename, fields, rows):
	# writing to csv file 
	with open(filename, 'w') as csvfile: 
		# creating a csv writer object
		csvwriter = csv.writer(csvfile)
			
		# writing the fields
		csvwriter.writerow(fields)
			
		# writing the data rows
		csvwriter.writerows(rows)

filename = "full_dataset.csv"
dataframe_dir = 'an2dl-homeworks/full_dataset/'
fields = ['image_id', 'label', 'filepath']
labels = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']

rows = create_csv_rows(dataframe_dir, labels)
write_csv_file(filename, fields, rows)
