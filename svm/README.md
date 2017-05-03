# svm.py

to run: ./svm.py <significance of f-test> <kernel type>
* <significance of f-test>: can be 0, 5, or 10. 0 uses all variables, 5 uses only variables that pass on 5% level of significance, and 10 only uses variable that pass 10% level of significance
* <kernel type>: can be rbf, poly, sigmoid, linear
** poly is a polynomial of degree 3
** rbf is radial bias function (gaussian)
