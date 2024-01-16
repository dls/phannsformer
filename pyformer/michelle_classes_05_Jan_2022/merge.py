class_names = ['HTJ', 'baseplate', 'HTJ_connector', 'other', 'baseplate_upper', 'major_capsid', 'collar_tail_fiber', 'major_tail', 'portal', 'shaft', 'minor_tail', 'tail_spike']

files = [[x+f'_{i}.fasta' for i in range(1,12)] for x in class_names]

for i in range(len(class_names)):
	file_group = files[i]
	class_name = class_names[i]

	with open(class_name + '_combined.fasta', 'a') as out:
		for file in file_group:
			with open(file, 'r') as f:
				f = f.read()
			out.write(f)
	del out


