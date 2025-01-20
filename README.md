# For OptBenchmarksLibrary development: 
- Please put all edits in "OptBenchmarksLibrary" folder and follow the code structure in it.

# ToDos (The orders of all bullet points are the priority)

## 0. Understand the framework
Look at these files to see how the general problem setup works & see examples:
- OptBenchmarksLibrary/Library_Test.ipynb
- OptBenchmarksLibrary/base.py
- OptBenchmarksLibrary/config.py
- Functions that are done (other functions I think still use the old version of set up):
  - Ackley and others in the Synthetic folder
  - Example functions in LassoBench
  - BraninCurrin in BoTorch folder
  - BBOB

```
code
```

Essentially, when you implement each function now, you no longer have to scale it at the very bottom level.
The only thing you'll have to consider is:
- Set up the correct configs for constraints, multiobjective, or mix-integer problems!
- Passing in the correct bounds
- Return GX and FX (GX is None only for unconstrained functions)
 
Therefore, when testing each function:
- For every function: make sure the function value are the EXACT SAME as the paper/library
- For constrained problems: test if the 3 different configs ALL works well
- For multiobjective problems: test if 4 different configs ALL works well
- (At this stage we should consider if writing test cases / test files are needed)

## 1. Add these functions
- Write all your previous functions in this framework format
  - For **engineering** problems: Put them under the engineering folder. The default config for engineering constraints are CONSTRANTS (return GX)
  For **Trusses**, ensure I can select which "versions." I have noticed that some Trusses functions in your previous implementation do not support different instances (let's use instances instead of versions). Double-check that the evaluation is returning the same values as listed in the paper.
  - For **MODACT**: implement all 20 (?) function Cyril has in his library. Be careful that most functions are multiobjective so implement as they are as the original setting
  - For **other** previously written functions: do not implement the embedding ones and ignore finders
- Translate all the CEC2020 code
  - Look at the .m files, they are the matlab source code
  - The problem definition .pdf have the problem formulation
- Add MOPTA and MAZDA car problems
- Make sure that the implementation for these libraries are completed:
  - LassoBench
  - Read "Scalable Global Optimization via Local Bayesian Optimization" and implement
    - Rover (I have the source code in the old version. Wrap it with the new format.

## 2. Make a sphinx website for the library
- Rosen need to upload the sphinx docs
- Official guilde: https://www.sphinx-doc.org/en/master/
- What we ultimately want to have is something like openmdao: https://openmdao.org/newdocs/versions/latest/main.html
 
