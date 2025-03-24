# March 24 Notes:
Hi hope you are doing well! Overall, great job on making good progress on literally everything! 

To-Dos:
1. Remove cont_to_disc: I'm unsure if you recognized a function called cont_to_disc in some mixed variable functions. So we want to remove this in every function. This library ONLY takes in raw data. AKA if the function can only take in discrete X values, we should not assume that the users will pass in continuous ones. (As cont_to_disc is technically a design choice in algorithms). Therefore, I want you to double check all the function ONLY assume that the X got passed in is **Correct**, meaning that if a feature is discrete, the user should pass in discrete X, and not continuous. 

2. Documentation: Update the documentation with all your additional functions and add tutorials for all categories of functions
  
3. More functions
   a. Trusses related function:
   b. Other source: Make sure our library includes the functions from these sources
   
   



# Feb 5 Notes
1. The OptBenchmarksLibrary/base.py have the BenchmarkProblem class
2. The actual f(x) in the most commonly used continuous version should be be a subclass of BenchmarkProblem
3. The other variant of f(x) should be a subclass of the actual f(x)
