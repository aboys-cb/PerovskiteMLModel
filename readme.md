# Introduction
This program integrates two machine learning models and supports the mixing of specified elements and their proportions, enabling high-throughput screening for potential perovskite materials.

## Basic Example
In the `main.py` file, you are required to define the element list for each site, which includes A, B, B1, and X. When the B site is occupied by divalent cations, B can be equivalent to B1. This distinction for the B site is primarily to differentiate between monovalent and trivalent ions.
When invoking the `random_perovskites` function, the `doping_ratio` parameter must be passed.
The `doping_ratio` is a nested list containing four elements, formatted as `doping_ratio=[A, B, B1, X]`. The proportion for each site is represented by a list; for instance, A=[0.5,0.5] indicates that the elements at the A site are mixed in a 50% ratio. The proportions must adhere to the following rules: the sum of elements in the A list equals 1; the sum of elements in the B and B1 lists equals 1; the sum of elements in the X list equals 3.
For example, `doping_ratio=[[1], [0.5], [0.25, 0.125, 0.125], [3]]` implies that the elements at the B site are mixed at a minimum ratio of 1/8.
  
When the `_filter=True` setting is applied, the program will directly assess the formability of materials based on ionic radii, returning only the results that meet the criteria. Enabling this option will significantly reduce the search space.

# Requirements
To prevent interface failures due to dependency library version updates, we recommend using Python 3.10.

`pip install -r requirements.txt`