# Installation
Hi hope you are doing well! Overall, great job on making good progress on literally everything! 

To-Dos:
1. Remove cont_to_disc: I'm unsure if you recognized a function called cont_to_disc in some mixed variable functions. So we want to remove this in every function. This library ONLY takes in raw data. AKA if the function can only take in discrete X values, we should not assume that the users will pass in continuous ones. (As cont_to_disc is technically a design choice in algorithms). Therefore, I want you to double check all the function ONLY assume that the X got passed in is **Correct**, meaning that if a feature is discrete, the user should pass in discrete X, and not continuous. 

2. Documentation: Update the documentation with all your additional functions and add tutorials for all categories of functions
  
3. More functions
  - Trusses related function:
    - Look into the other folder and find the truss problems. In theory, these files should also include the paper on the truss problems (let me know if you don't have access to the papers). Look at the original problem formulated in the paper and check if the implementation on our side is correct since the python library for this truss problem does not specify the units of anything.
    - From the software perspective, I think some Truss problems have different variances, think about how to best structure them. Right now we classify them as Truss10D, 25D, ...etc.
  - Other source: Make sure our library includes all the problems / functions from these sources、
    - NEORL: https://neorl.readthedocs.io/en/latest/index.html
    - The real-world problem from this: https://dl.acm.org/doi/10.1145/3583133.3595060 (Let me know if you have access)
   
   



# Feb 5 Notes
1. The OptBenchmarksLibrary/base.py have the BenchmarkProblem class
2. The actual f(x) in the most commonly used continuous version should be be a subclass of BenchmarkProblem
3. The other variant of f(x) should be a subclass of the actual f(x)
