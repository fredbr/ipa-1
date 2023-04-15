run_test:
	./run-codes-local/grade-single.py -e -f 0.1 -d assignments/$(name)/test_cases -c assignments/$(name)/src/$(name).py

run_case:
	cd assignments/$(name)/test_cases &&\
	/usr/bin/env python3 ./../src/$(name).py < $(case) 