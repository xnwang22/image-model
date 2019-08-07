# Define required macros here
train:
	python python/tf_john-matt.py
capture:
	python python/capture2.py -n ${name} -t ${target} -c ${count}