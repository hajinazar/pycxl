# **pycxl: Python Wrapper for Computing the Convex Hull**

*pycxl* (version 1.6) is a Python script that:

* computes the convex hull for data points of a system of arbitrary 
  dimensions (e.g., binary, ternary, quaternary, etc.),

* calculates and outputs the “distance above hull” for all data points,

* produces output files suitable for creating convex hull plots for
  binary and ternary systems,

* creates a 2-D plot output for binaries and ternaries if the matplotlib
  module exists on the system.

## **Dependencies**

The *pycxl* script requires *scipy* and *numpy* modules and -optionally-
uses *matplotlib* to produce convex hull plots for binary and ternary system.

## **Usage**

The script can be used as:

```bash
pycxl.py input_file
```

where the *input_file* is a text file with the input data. If the input file
is not specified, the script will look for the *points.txt* file in the
working directory to read the input data.

## **How to use?**

See the howto.pdf file.

## **Contact and bug reports**

Samad Hajinazar <br />
Department of Chemistry, State University of New York at Buffalo <br />
samadh\~at\~buffalo.edu
