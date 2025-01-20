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

The main change is the .evaluate() function
```
def evaluate(self, X):
        # Process input
        X = self.INPUT_TYPE_CONVERT(X)
        
        if self.MIXED.is_mixed:
            X = self.mixed_int_scale(X)
        else:
            X = self.scale(X)
        
        # Call the actual evaluation implementation
        gx, fx = self._evaluate_implementation(X)

        # Process Constraints
        gx, fx = self.constraint_processing(gx, fx)

        # Process if MultiObjective
        fx = self.multiobj_processing(fx)
        
        # Process output type
        fx = self.OUTPUT_TYPE_CONVERT(fx)
        if gx != None:
            gx = self.OUTPUT_TYPE_CONVERT(gx)

        # Negate for minimization setting
        if not self.MAXIMIZATION:
            fx = -fx
        
        return gx, fx
```

Essentially, what we are implementing is the `_evaluate_implementation` function. When you implement each function now, you no longer have to scale it at the very bottom level.
The only thing you'll have to consider is:
- Set up the correct configs for constraints, multiobjective, or mix-integer problems!
- Passing in the correct bounds
- Return GX and FX (GX is None only for unconstrained functions)
 
Therefore, when testing each function:
- For every function: make sure the function values are the EXACT SAME as the paper/library
- For constrained problems: test if the 3 different configs ALL works well
- For multiobjective problems: test if 4 different configs ALL works well
- (At this stage, we should maybe(?) consider if writing test cases / test files are needed)

## 1. Add these functions
- Go through all functions I put in `Synthetics`, `Engineering`, `LassoBench`, `BoTorch`, and `BBOB` that they work in this format (since I think put some of the old version there)
- Put all the previous functions in your folder in this framework format
  - For **engineering** problems: Put them under the engineering folder. The default config for engineering constraints are CONSTRANTS (return GX)
  For **Trusses**, ensure I can select which "versions." I have noticed that some Trusses functions in your previous implementation do not support different instances (let's use instances instead of versions). Double-check that the evaluation is returning the same values as listed in the paper.
  - For **MODACT**: implement all 20 (?) function Cyril has in his library. Be careful that most functions are multiobjective so implement as they are as the original setting
  - For **other** previously written functions: do not implement the embedding ones and ignore finders
- Translate all the CEC2020 code 
  - Look at the .m files, they are the matlab source code
  - The problem definition .pdf have the problem formulation
- Add MOPTA and MAZDA car problems (put them under engineering)
  - Sources of my original code for MOPTA and MAZDA (see or download this first):
    - Mazda: https://www.dropbox.com/scl/fo/2mdmipyasdt9bgqcnagc6/ANzv5eRr5o4b5ZPwOsUDGMc?rlkey=4gsukfu7tz0znhsze6878l3h7&st=8il4r2o0&dl=0
    - Test functions for all: https://www.dropbox.com/scl/fo/flpvnpkkiid7nojrdmuo3/ANi_64jVXO3FLmU8vtxbdNI?rlkey=jch7dupoi4bsari1ggdus23uq&st=hisdyh6l&dl=0
  - For MOPTA source:
    - https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html
  - For MAZDA:
    - https://ladse.eng.isas.jaxa.jp/benchmark/
- Make sure that the implementation for these libraries are completed:
  - Read "Scalable Global Optimization via Local Bayesian Optimization" by David Eriksson et al. and implement
    - Rover (I have the source code in the old version. Wrap it with the new format)
    - Robot pushing
    - Cosmological constant learning
    - And other problems that look like engineering
    - Sources:
      - https://github.com/uber-research/TuRBO
      - https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/4e6f9ed04833cc2e21b5906b1181bc067298f914
        - so you can also read this paper: https://arxiv.org/pdf/1706.01445
  - (After everything is done until this bullet point, I think we are good to publish the library)
  - LassoBench (I implement like 3 of them??)
  - Make sure all bbob suite work perfect:
    - https://coco-platform.org/
    - https://numbbo.github.io/coco/testsuites/bbob
  - All Botorch functions in the botorch/test_functions
  - All engineering problems in pymoo
  - All Real-World Problems in "GECCO 2023 Tutorial on Benchmarking Multiobjective Optimizers 2.0"
    - https://dl.acm.org/doi/abs/10.1145/3583133.3595060?casa_token=jYey2h3Kcn0AAAAA:Ko_vDbjT-9aEGxCLvsAy8XZcbDQP05sUKwvoO0PVVm61nWb3LK6AKMFGzMX17wgUlQDpiyNdRbrC5w

## 2. Make a sphinx website for the library (work in parallel with 1.)
- The current website is in "OptBench_docs" folder
  - You'll need to install sphinx and rebuild the file (try `make html` or start from building: https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)
- Official guide for using sphinx: https://www.sphinx-doc.org/en/master/
- What we ultimately want to have is something like openmdao: https://openmdao.org/newdocs/versions/latest/main.html
- Please update this every week up to date as you implement each function
- Does not have to be long, just provide a 1 or 2 line sentence summary of what each problem is about and give an example how to use each problem's code (aka what configs) in their page
  - If they are coming from some source, put the original source link and cite it
  - If not, write the formulation of the problem in latex
 
# Let's try to finish this by the end of this month!!
