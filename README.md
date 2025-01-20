# For OptBenchmarksLibrary development: 
- Please put all edits in "OptBenchmarksLibrary" folder and follow the code structure in it.

# ToDos (The orders of all bullet points are the priority)

## 0. Understand the framework
Look at these files to see how the general problem setup work & see examples:
- OptBenchmarksLibrary/Library_Test.ipynb
- OptBenchmarksLibrary/base.py
- OptBenchmarksLibrary/config.py
- Functions that are done (other functions I think still use the old version of set up):
  - Ackley and others in the Synthetic folder
  - Example functions in LassoBench
  - BraninCurrin in BoTorch folder
  - BBOB
 
Therefore, when testing each function:
- For every function: make sure the function value are the EXACT SAME as the paper/library
- For constrained problems: test if the 3 different configs ALL works well
- For multiobjective problems: test if 4 different configs ALL works well
- (At this stage we should consider if writing test cases / test files are needed)

## 1. Add these functions
- Write all your functions in this frameworks format
  - For **engineering** problems: Put them under the engineering folder. The default config for engineering constraints are CONSTRANTS (return GX)
  - For **Trusses**: ensure I can select which "versions". I have noticed that there are some Trusses functions in your previous implementation that do not support different instances (let's use instances and not versions). Double-check if the evaluation is returning the same values as what is listed in the paper.
  - For **MODACT**: implement all 20 (?) function Cyril has in his library. Be careful that most functions are multiobjective so implement as they are as the original setting
  - For **other** previously written functions: ignore the embedding ones and ignore finders
  

## 2. Make a sphinx website for the library
- Rosen need to upload the sphinx docs
- Official guilde: https://www.sphinx-doc.org/en/master/
- What we ultimately want to have is something like openmdao: https://openmdao.org/newdocs/versions/latest/main.html
 
