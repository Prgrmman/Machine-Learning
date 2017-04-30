#!/usr/bin/bash



function testLog(){
	values=(0.001 0.05 0.1 0.5)
	for i in "${values[@]}"
	do
		./logistic.py $1 $i $2
	done
}

printf "\t===Running tests on middle east data===\n"
testLog middle_east all



printf "\t===Running tests on math===\n"
printf "*** Using all features ***\n"
testLog por_math all
printf "*** Using f-test 10 percent sig-level features ***\n"
testLog por_math 10
printf "*** Using f-test 5 percent sig-level features ***\n"
testLog por_math 5

printf "\n\t===Running tests on language===\n"
testLog por_lang all
printf "*** Using f-test 10 percent sig-level features ***\n"
testLog por_lang 10
printf "*** Using f-test 5 percent sig-level features ***\n"
testLog por_lang 5

