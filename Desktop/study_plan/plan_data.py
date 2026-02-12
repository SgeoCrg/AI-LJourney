
# plan_data.py

PLAN_DATA = {
    "title": "Become a Data Scientist in 7 Months - Detailed Study Plan",
    "modules": [
        {
            "name": "Module 1: Statistics & Core Data Science Techniques",
            "duration": "Weeks 1-6 (Month 1 & half of Month 2)",
            "weeks": [
                {
                    "week_num": 1,
                    "theme": "Python Basics for Data Science",
                    "days": [
                        {
                            "day": "Day 1", 
                            "topic": "Python Setup & Basics", 
                            "theory": """
                            <b>Environment Setup:</b>
                            <br/>1. <b>Install Anaconda:</b> Download the Individual Edition from <a href="https://www.anaconda.com/products/individual" color="blue">anaconda.com</a>. This installs Python, Jupyter, and key libraries.
                            <br/>2. <b>Jupyter Notebook:</b> A web-based interactive interface. Launch it from Anaconda Navigator or terminal (`jupyter notebook`).
                            <br/>
                            <br/><b>Core Concepts:</b>
                            <br/>- <b>Comments:</b> Use `#` for single-line comments.
                            <br/>- <b>Print:</b> `print("Hello")` displays text.
                            <br/>- <b>Math Operators:</b> `+`, `-`, `*`, `/`, `**` (exponent), `%` (modulus).
                            <br/>
                            <br/><b>Resources:</b>
                            <br/>- <a href="https://docs.python.org/3/tutorial/interpreter.html" color="blue">Python Docs: Using the Interpreter</a>
                            <br/>- <a href="https://realpython.com/jupyter-notebook-introduction/" color="blue">Real Python: Jupyter Notebook Intro</a>
                            """, 
                            "practice": """
                            1. Install Anaconda and launch Jupyter Notebook.
                            2. Create a new notebook named `Day1_Basics`.
                            3. In the first cell, type `print("Hello, World!")` and run it (Shift+Enter).
                            4. In a new cell, calculate `2 + 2`, `10 / 2`, and `2 ** 3`.
                            5. Create a variable `message = "My first script"` and print it.
                            """, 
                            "time": "2.5h"
                        },
                        {
                            "day": "Day 2", 
                            "topic": "Variables & Data Types", 
                            "theory": """
                            <b>Core Concepts:</b>
                            <br/>- <b>Variables:</b> Containers for storing data values. Created upon assignment (e.g., x = 5).
                            <br/>- <b>Data Types:</b>
                            <br/>  * Text Type: str
                            <br/>  * Numeric Types: int, float, complex
                            <br/>  * Sequence Types: list, tuple, range
                            <br/>  * Mapping Type: dict
                            <br/>  * Set Types: set, frozenset
                            <br/>  * Boolean Type: bool
                            <br/>  * Binary Types: bytes, bytearray, memoryview
                            <br/>
                            <br/><b>Resources:</b>
                            <br/>- <a href="https://realpython.com/python-variables/" color="blue">Real Python: Variables</a>
                            <br/>- <a href="https://www.w3schools.com/python/python_variables.asp" color="blue">W3Schools: Python Variables</a>
                            """, 
                            "practice": """
                            1. Create a variable `x` and assign it an integer.
                            2. Create a variable `y` and assign it a float.
                            3. Create a variable `name` and assign it your name (string).
                            4. Print the type of each variable using `type()`.
                            5. Convert `x` to a float and `y` to an integer.
                            6. Create a complex number and print its real and imaginary parts.
                            """, 
                            "time": "2.5h"
                        },
                        {
                            "day": "Day 3", 
                            "topic": "Control Flow (If/Else, Loops)", 
                            "theory": """
                            <b>Conditionals:</b>
                            <br/>- `if`, `elif`, `else` blocks allow code to make decisions.
                            <br/>- <b>Indentation:</b> Critical in Python! Blocks are defined by indentation (4 spaces).
                            <br/>
                            <br/><b>Loops:</b>
                            <br/>- <b>For Loops:</b> Iterate over a sequence. `for i in range(5):`
                            <br/>- <b>While Loops:</b> Repeat while a condition is true. `while x < 5:`
                            <br/>- <b>Control Keywords:</b> `break` (exit loop), `continue` (skip iteration).
                            <br/>
                            <br/><b>Resources:</b>
                            <br/>- <a href="https://docs.python.org/3/tutorial/controlflow.html" color="blue">Python Docs: Control Flow</a>
                            <br/>- <a href="https://www.w3schools.com/python/python_for_loops.asp" color="blue">W3Schools: For Loops</a>
                            """, 
                            "practice": """
                            1. <b>Even/Odd Checker:</b> Write a script that checks if a variable `n` is even or odd using `if/else` and `%`.
                            2. <b>Looping:</b> Print numbers from 1 to 10 using a `for` loop.
                            3. <b>FizzBuzz:</b> Print numbers 1-20. If divisible by 3, print "Fizz". If by 5, "Buzz". If both, "FizzBuzz".
                            4. <b>Summation:</b> Calculate the sum of all numbers from 1 to 100 using a loop.
                            """, 
                            "time": "2.5h"
                        },
                        {
                            "day": "Day 4", 
                            "topic": "Functions & Modules", 
                            "theory": """
                            <b>Functions:</b>
                            <br/>- Reusable blocks of code defined with `def`.
                            <br/>- <b>Parameters:</b> Inputs to the function. `def greet(name):`
                            <br/>- <b>Return:</b> Sends a result back. `return x + y`
                            <br/>
                            <br/><b>Modules:</b>
                            <br/>- Files containing Python code (functions, variables).
                            <br/>- Import using `import module_name` or `from module import func`.
                            <br/>- Standard Library: `math`, `random`, `datetime`.
                            <br/>
                            <br/><b>Resources:</b>
                            <br/>- <a href="https://www.w3schools.com/python/python_functions.asp" color="blue">W3Schools: Functions</a>
                            <br/>- <a href="https://realpython.com/python-modules-packages/" color="blue">Real Python: Modules</a>
                            """, 
                            "practice": """
                            1. Write a function `calculate_area(radius)` that returns the area of a circle. (Use `math.pi`).
                            2. Write a function `celsius_to_fahrenheit(c)` that converts temp.
                            3. Create a new file `my_utils.py`, put your functions there.
                            4. In your main notebook, `import my_utils` and use the functions to calculate the area of a circle with radius 5.
                            """, 
                            "time": "2.5h"
                        },
                        {
                            "day": "Day 5", 
                            "topic": "Data Structures (Lists, Dicts)", 
                            "theory": """
                            <b>Lists (`[]`):</b>
                            <br/>- Ordered, mutable sequences.
                            <br/>- Methods: `.append()`, `.pop()`, `.sort()`.
                            <br/>- Slicing: `my_list[start:end]`.
                            <br/>
                            <br/><b>Dictionaries (`{}`):</b>
                            <br/>- Key-Value pairs. Unordered (pre-3.7).
                            <br/>- Access: `my_dict['key']`.
                            <br/>- Methods: `.keys()`, `.values()`, `.items()`.
                            <br/>
                            <br/><b>Resources:</b>
                            <br/>- <a href="https://docs.python.org/3/tutorial/datastructures.html" color="blue">Python Docs: Data Structures</a>
                            <br/>- <a href="https://realpython.com/python-lists-tuples/" color="blue">Real Python: Lists & Tuples</a>
                            """, 
                            "practice": """
                            1. <b>Lists:</b> Create a list of 5 fruits. Add a 6th. Remove the 2nd. Print the last 3.
                            2. <b>Dicts:</b> Create a `student` dictionary with keys: name, age, grades (a list of ints).
                            3. Print the average grade of the student from the dictionary.
                            4. Create a list of dictionaries representing 3 students. Loop through and print their names.
                            """, 
                            "time": "2.5h"
                        },
                        {
                            "day": "Day 6", 
                            "topic": "Week 1 Review & Mini-Project", 
                            "theory": """
                            <b>Consolidation:</b>
                            <br/>Review variables, loops, conditionals, functions, and data structures.
                            <br/>
                            <br/><b>Project Logic:</b>
                            <br/>- Break down the problem into small steps.
                            <br/>- Write pseudocode before actual code.
                            <br/>- Test frequently (print debugging).
                            """, 
                            "practice": """
                            <b>Project: Number Guessing Game</b>
                            1. Import `random` and generate a secret number (1-100).
                            2. Use a `while` loop to ask user for input: `input("Guess: ")`.
                            3. Convert input to int.
                            4. If guess == secret, print "You won!" and break.
                            5. If guess < secret, print "Too low". Else, "Too high".
                            6. Bonus: Limit the number of attempts to 7.
                            """, 
                            "time": "3h"
                        },
                        {
                            "day": "Day 7", 
                            "topic": "Rest Day", 
                            "theory": """
                            <b> The Science of Rest:</b>
                            <br/>- <b>Memory Consolidation:</b> Your brain solidifies neural pathways formed during the week while you rest.
                            <br/>- <b>Prevent Burnout:</b> Consistency > Intensity. Taking a break ensures you can start Week 2 fresh.
                            """, 
                            "practice": """
                            1. Close your laptop.
                            2. Go for a walk, exercise, or read a non-technical book.
                            3. Do NOT write any code today.
                            """, 
                            "time": "0h"
                        }
                    ]
                },
                {
                    "week_num": 2,
                    "theme": "NumPy & Pandas Basics",
                    "days": [
                        {"day": "Day 8", "topic": "Intro to NumPy Arrays", "theory": "https://numpy.org/doc/stable/user/absolute_beginners.html", "practice": "Create arrays, perform element-wise arithmetic, reshaping.", "time": "2.5h"},
                        {"day": "Day 9", "topic": "NumPy Indexing & Broadcasting", "theory": "https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html", "practice": "Matrix multiplication and array slicing exercises.", "time": "2.5h"},
                        {"day": "Day 10", "topic": "Intro to Pandas DataFrames", "theory": "https://pandas.pydata.org/docs/user_guide/10min.html", "practice": "Load a CSV file. Inspect head(), info(), describe().", "time": "2.5h"},
                        {"day": "Day 11", "topic": "Pandas Selection & Filtering", "theory": "https://pandas.pydata.org/docs/user_guide/indexing.html", "practice": "Filter rows based on conditions (e.g. titanic dataset survivors).", "time": "2.5h"},
                        {"day": "Day 12", "topic": "Pandas Missing Data & Cleaning", "theory": "https://pandas.pydata.org/docs/user_guide/missing_data.html", "practice": "Handle NaN values (fillna, dropna) in a messy dataset.", "time": "2.5h"},
                        {"day": "Day 13", "topic": "Review & Week 2 Mini-Project", "theory": "Review NumPy/Pandas docs", "practice": "Analyze the 'Titanic' dataset on Kaggle (basic stats).", "time": "3h"},
                        {"day": "Day 14", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 3,
                    "theme": "Data Visualization",
                    "days": [
                        {"day": "Day 15", "topic": "Matplotlib Basics", "theory": "https://matplotlib.org/stable/tutorials/introductory/quick_start.html", "practice": "Plot line charts, bar charts, and histograms from random data.", "time": "2.5h"},
                        {"day": "Day 16", "topic": "Subplots & Customization", "theory": "https://realpython.com/python-matplotlib-guide/", "practice": "Create a figure with 4 subplots (scatter, line, bar, hist) with titles/labels.", "time": "2.5h"},
                        {"day": "Day 17", "topic": "Seaborn Introduction", "theory": "https://seaborn.pydata.org/tutorial/introduction.html", "practice": "Recreate Matplotlib plots using Seaborn (sns.scatterplot, sns.histplot).", "time": "2.5h"},
                        {"day": "Day 18", "topic": "Categorical & Distribution Plots", "theory": "https://seaborn.pydata.org/tutorial/categorical.html", "practice": "Use boxplots and violin plots to visualize distributions.", "time": "2.5h"},
                        {"day": "Day 19", "topic": "Heatmaps & Correlation", "theory": "https://seaborn.pydata.org/examples/many_pairwise_correlations.html", "practice": "Compute correlation matrix of a dataset and plot a heatmap.", "time": "2.5h"},
                        {"day": "Day 20", "topic": "Exploratory Data Analysis (EDA) Project", "theory": "Reading: 'A Gentle Introduction to EDA' (MachineLearningMastery)", "practice": "Perform full EDA on the 'Iris' dataset (pairplots, stats).", "time": "3h"},
                        {"day": "Day 21", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 4,
                    "theme": "Statistics Fundamentals",
                    "days": [
                        {"day": "Day 22", "topic": "Descriptive Statistics", "theory": "Khan Academy: Descriptive Statistics", "practice": "Calculate mean, median, mode, variance, std dev in Python (scipy.stats).", "time": "2.5h"},
                        {"day": "Day 23", "topic": "Probability Basics", "theory": "Khan Academy: Probability", "practice": "Simulate coin flips and dice rolls in Python to understand probability.", "time": "2.5h"},
                        {"day": "Day 24", "topic": "Probability Distributions (Normal, Binomial)", "theory": "StatQuest: The Normal Distribution", "practice": "Plot Normal and Binomial distributions using Scipy/Matplotlib.", "time": "2.5h"},
                        {"day": "Day 25", "topic": "Hypothesis Testing (t-tests)", "theory": "StatQuest: Hypothesis Testing/p-values", "practice": "Perform a t-test on two groups of data using scipy.stats.ttest_ind.", "time": "2.5h"},
                        {"day": "Day 26", "topic": "Correlation vs Causation & ANOVA", "theory": "StatQuest: ANOVA", "practice": "Check correlation (Pearson/Spearman) and run one-way ANOVA.", "time": "2.5h"},
                        {"day": "Day 27", "topic": "Stats Mini-Project", "theory": "Review Stats concepts", "practice": "Analyze 'Housing Prices' dataset for statistical significance of features.", "time": "3h"},
                        {"day": "Day 28", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                 {
                    "week_num": 5,
                    "theme": "Unsupervised Learning: Clustering",
                    "days": [
                        {"day": "Day 29", "topic": "Intro to Unsupervised Learning", "theory": "Scikit-Learn User Guide: Clustering", "practice": "Read documentation and understand the difference from Supervised Learning.", "time": "2.5h"},
                        {"day": "Day 30", "topic": "K-Means Clustering", "theory": "StatQuest: K-Means Clustering", "practice": "Implement K-Means on the Iris dataset. Visualize clusters.", "time": "2.5h"},
                        {"day": "Day 31", "topic": "Choosing K (Elbow Method)", "theory": "Medium: Elbow Method for K-Means", "practice": "Plot inertia vs K to find optimal clusters for a dataset.", "time": "2.5h"},
                        {"day": "Day 32", "topic": "Hierarchical Clustering", "theory": "StatQuest: Hierarchical Clustering", "practice": "Generate strict dendrograms using scipy.cluster.hierarchy.", "time": "2.5h"},
                        {"day": "Day 33", "topic": "DBSCAN", "theory": "Scikit-Learn User Guide: DBSCAN", "practice": "Cluster noisy data (make_moons) using DBSCAN vs K-Means.", "time": "2.5h"},
                        {"day": "Day 34", "topic": "Clustering Project", "theory": "-", "practice": "Segment customers based on 'Mall Customers' dataset (Kaggle).", "time": "3h"},
                        {"day": "Day 35", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                 {
                    "week_num": 6,
                    "theme": "Dimensionality Reduction",
                    "days": [
                        {"day": "Day 36", "topic": "Curse of Dimensionality & PCA Theory", "theory": "StatQuest: PCA (Principal Component Analysis)", "practice": "Watch video and summarize key points.", "time": "2.5h"},
                        {"day": "Day 37", "topic": "Implementing PCA in Python", "theory": "Scikit-Learn User Guide: PCA", "practice": "Apply PCA to the Iris dataset and visualize 2D projection.", "time": "2.5h"},
                        {"day": "Day 38", "topic": "Explained Variance", "theory": "Explained Variance Ratio in PCA", "practice": "Plot cumulative explained variance to choose n_components.", "time": "2.5h"},
                        {"day": "Day 39", "topic": "t-SNE & UMAP (Visualization)", "theory": "StatQuest: t-SNE", "practice": "Visualize partial MNIST digits using t-SNE.", "time": "2.5h"},
                        {"day": "Day 40", "topic": "Feature Selection Techniques", "theory": "Scikit-Learn: Feature Selection", "practice": "Use SelectKBest and VarianceThreshold on a dataset.", "time": "2.5h"},
                        {"day": "Day 41", "topic": "Module 1 Capstone", "theory": "Review Module 1", "practice": "End-to-End EDA + Clustering + PCA on a new dataset (e.g. Wine Quality).", "time": "3h"},
                        {"day": "Day 42", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                }
            ]
        },
        {
            "name": "Module 2: Solving Business Problems with Supervised Learning",
            "duration": "Weeks 7-12 (Month 2 [part] - Month 3)",
            "weeks": [
                {
                    "week_num": 7,
                    "theme": "Linear Regression",
                    "days": [
                        {"day": "Day 43", "topic": "Simple Linear Regression", "theory": "StatQuest: Linear Regression", "practice": "Implement Simple LR from scratch (using specific formulas) then check with sklearn.", "time": "2.5h"},
                        {"day": "Day 44", "topic": "Multiple Linear Regression", "theory": "Scikit-Learn: Linear Models", "practice": "Predict housing prices using multiple features.", "time": "2.5h"},
                        {"day": "Day 45", "topic": "Regression Metrics (MSE, RMSE, R2, MAE)", "theory": "Medium: Regression Metrics Explained", "practice": "Calculate these metrics manually and using sklearn.metrics.", "time": "2.5h"},
                        {"day": "Day 46", "topic": "Polynomial Regression", "theory": "Towards Data Science: Polynomial Regression", "practice": "Fit a non-linear trend using PolynomialFeatures.", "time": "2.5h"},
                        {"day": "Day 47", "topic": "Regularization (Lasso/Ridge)", "theory": "StatQuest: Ridge vs Lasso", "practice": "Apply Ridge/Lasso to reduce overfitting on a noisy dataset.", "time": "2.5h"},
                        {"day": "Day 48", "topic": "Regression Project", "theory": "-", "practice": "Kaggle: House Prices - Advanced Regression Techniques (Basic submission).", "time": "3h"},
                        {"day": "Day 49", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 8,
                    "theme": "Classification Basics",
                    "days": [
                        {"day": "Day 50", "topic": "Logistic Regression Theory", "theory": "StatQuest: Logistic Regression", "practice": "Understand Logit function and decision boundaries.", "time": "2.5h"},
                        {"day": "Day 51", "topic": "Implementing Logistic Regression", "theory": "Scikit-Learn: Logistic Regression", "practice": "Classify Iris (Setosa vs Versicolor) using Logistic Regression.", "time": "2.5h"},
                        {"day": "Day 52", "topic": "Classification Metrics (Accuracy, Precision, Recall)", "theory": "Scikit-Learn: Classification Metrics", "practice": "Compute confusion matrix and metrics for your model.", "time": "2.5h"},
                        {"day": "Day 53", "topic": "ROC Curve & AUC", "theory": "StatQuest: ROC and AUC", "practice": "Plot ROC curve and calculate AUC score.", "time": "2.5h"},
                        {"day": "Day 54", "topic": "K-Nearest Neighbors (KNN)", "theory": "StatQuest: K-Nearest Neighbors", "practice": "Implement KNN Classification and tune 'k'.", "time": "2.5h"},
                        {"day": "Day 55", "topic": "Classification Mini-Project", "theory": "-", "practice": "Predict Survival on Titanic (Binary Classification).", "time": "3h"},
                        {"day": "Day 56", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 9,
                    "theme": "Support Vector Machines (SVM)",
                    "days": [
                        {"day": "Day 57", "topic": "SVM Intuition & Margins", "theory": "StatQuest: Support Vector Machines", "practice": "Visualizing Hyperplanes and Support Vectors.", "time": "2.5h"},
                        {"day": "Day 58", "topic": "Kernels (Linear, RBF, Poly)", "theory": "StatQuest: The Kernel Trick", "practice": "Test different kernels on non-linearly separable data (make_circles).", "time": "2.5h"},
                        {"day": "Day 59", "topic": "SVM Hyperparameters (C, Gamma)", "theory": "Scikit-Learn Docs: SVM", "practice": "Use GridSearch to find best C and Gamma.", "time": "2.5h"},
                        {"day": "Day 60", "topic": "SVM for Regression (SVR)", "theory": "Medium: SVR Explained", "practice": "Apply SVR to a regression problem (e.g. Boston Housing).", "time": "2.5h"},
                        {"day": "Day 61", "topic": "Review SVM", "theory": "-", "practice": "Compare SVM vs Logistic Regression performance.", "time": "2.5h"},
                        {"day": "Day 62", "topic": "Weekly Challenge", "theory": "-", "practice": "Classify Handwritten Digits (load_digits) using SVM.", "time": "3h"},
                        {"day": "Day 63", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 10,
                    "theme": "Decision Trees & Random Forests",
                    "days": [
                        {"day": "Day 64", "topic": "Decision Trees Theory (Entropy, Gini)", "theory": "StatQuest: Decision Trees", "practice": "Calculate Gini Impurity manually for a small split.", "time": "2.5h"},
                        {"day": "Day 65", "topic": "Implementing Decision Trees", "theory": "Scikit-Learn: Decision Trees", "practice": "Visualize a Decision Tree using graphviz/plot_tree.", "time": "2.5h"},
                        {"day": "Day 66", "topic": "Overfitting & Pruning", "theory": "Cost Complexity Pruning Path", "practice": "Train a deep tree vs a pruned tree (max_depth).", "time": "2.5h"},
                        {"day": "Day 67", "topic": "Random Forest Theory (Bagging)", "theory": "StatQuest: Random Forests", "practice": "Understand Bootstrap Aggregation.", "time": "2.5h"},
                        {"day": "Day 68", "topic": "Implementing Random Forests", "theory": "Scikit-Learn: RandomForestClassifier", "practice": "Train RF on a dataset and inspect feature_importances_.", "time": "2.5h"},
                        {"day": "Day 69", "topic": "Tree Project", "theory": "-", "practice": "Predict Heart Disease (UCI Dataset) using RF.", "time": "3h"},
                        {"day": "Day 70", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 11,
                    "theme": "Ensemble Methods (Boosting)",
                    "days": [
                        {"day": "Day 71", "topic": "Boosting Intuition (AdaBoost)", "theory": "StatQuest: AdaBoost", "practice": "Implement AdaBoostClassifier from sklearn.", "time": "2.5h"},
                        {"day": "Day 72", "topic": "Gradient Boosting", "theory": "StatQuest: Gradient Boost", "practice": "Train GradientBoostingClassifier.", "time": "2.5h"},
                        {"day": "Day 73", "topic": "XGBoost (Extreme Gradient Boosting)", "theory": "Documentation: XGBoost", "practice": "Install XGBoost and train a model. Compare speed with sklearn.", "time": "2.5h"},
                        {"day": "Day 74", "topic": "LightGBM & CatBoost", "theory": "Towards Data Science: LightGBM/CatBoost", "practice": "Install LightGBM. Train on large dataset.", "time": "2.5h"},
                        {"day": "Day 75", "topic": "Stacking & Voting Classifiers", "theory": "Scikit-Learn: VotingClassifier", "practice": "Combine LogReg, SVM, and RF into a VotingClassifier.", "time": "2.5h"},
                        {"day": "Day 76", "topic": "Ensemble Project", "theory": "-", "practice": "Kaggle: Predict Customer Churn using XGBoost.", "time": "3h"},
                        {"day": "Day 77", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 12,
                    "theme": "Model Validation & Tuning",
                    "days": [
                        {"day": "Day 78", "topic": "Cross-Validation Strategies", "theory": "Scikit-Learn: Cross Validation", "practice": "Implement K-Fold, Stratified K-Fold CV.", "time": "2.5h"},
                        {"day": "Day 79", "topic": "Hyperparameter Tuning (GridSearchCV)", "theory": "Scikit-Learn: Grid Search", "practice": "Tune an RF model using Grid Search.", "time": "2.5h"},
                        {"day": "Day 80", "topic": "RandomizedSearchCV", "theory": "Scikit-Learn: Randomized Search", "practice": "Tune XGBoost using Randomized Search (faster).", "time": "2.5h"},
                        {"day": "Day 81", "topic": "Pipelines in Scikit-Learn", "theory": "Scikit-Learn: Pipelines", "practice": "Build a pipeline with scaler -> PCA -> Classifier.", "time": "2.5h"},
                        {"day": "Day 82", "topic": "Handling Class Imbalance", "theory": "SMOTE (Synthetic Minority Over-sampling Technique)", "practice": "Use imblearn to apply SMOTE to an imbalanced dataset.", "time": "2.5h"},
                        {"day": "Day 83", "topic": "Module 2 Capstone", "theory": "Review Module 2", "practice": "Complete Supervised Learning Project on a complex dataset.", "time": "3h"},
                        {"day": "Day 84", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                }
            ]
        },
        {
            "name": "Module 3: Applying Advanced Data Science Techniques",
            "duration": "Weeks 13-22 (Months 4-5)",
            "weeks": [
                {
                    "week_num": 13,
                    "theme": "Introduction to Neural Networks",
                    "days": [
                        {"day": "Day 85", "topic": "The Perceptron", "theory": "StatQuest: Neural Networks Pt 1", "practice": "Implement a single perceptron from scratch.", "time": "2.5h"},
                        {"day": "Day 86", "topic": "Multi-Layer Perceptron (MLP)", "theory": "3Blue1Brown: But what is a Neural Network?", "practice": "Use sklearn MLPClassifier.", "time": "2.5h"},
                        {"day": "Day 87", "topic": "Activation Functions (Sigmoid, ReLU)", "theory": "DeepLearning.AI Notes", "practice": "Plot different activation functions.", "time": "2.5h"},
                        {"day": "Day 88", "topic": "Loss Functions & Backpropagation", "theory": "StatQuest: Backpropagation", "practice": "Understand the chain rule logic (conceptual).", "time": "2.5h"},
                        {"day": "Day 89", "topic": "Optimizers (SGD, Adam)", "theory": "Towards Data Science: Optimizers", "practice": "Experiment with different solvers in MLPClassifier.", "time": "2.5h"},
                        {"day": "Day 90", "topic": "Intro NN Project", "theory": "-", "practice": "Classify MNIST digits using MLPClassifier.", "time": "3h"},
                        {"day": "Day 91", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 14,
                    "theme": "Deep Learning with TensorFlow/Keras",
                    "days": [
                        {"day": "Day 92", "topic": "TensorFlow/Keras Setup & Basics", "theory": "TensorFlow Tutorials: Quickstart", "practice": "Install TF. Build a sequential model.", "time": "2.5h"},
                        {"day": "Day 93", "topic": "Building a Neural Net in Keras", "theory": "Keras Documentation", "practice": "Re-do MNIST classification using Keras Layers.", "time": "2.5h"},
                        {"day": "Day 94", "topic": "Training & Validation Splitting", "theory": "TF Guide: Training & Evaluation", "practice": "Use validation_split and plot history (acc/loss).", "time": "2.5h"},
                        {"day": "Day 95", "topic": "Early Stopping & Model Checkpoints", "theory": "Keras Callbacks API", "practice": "Implement EarlyStopping to prevent overfitting.", "time": "2.5h"},
                        {"day": "Day 96", "topic": "Dropout & Regularization in DL", "theory": "Journal of ML Research (Dropout)", "practice": "Add Dropout layers to your model.", "time": "2.5h"},
                        {"day": "Day 97", "topic": "Keras Project", "theory": "-", "practice": "Predict Housing Prices (Regression with Keras).", "time": "3h"},
                        {"day": "Day 98", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 15,
                    "theme": "Computer Vision (CNNs)",
                    "days": [
                        {"day": "Day 99", "topic": "Convolution Operation", "theory": "StatQuest: Convolutional Neural Networks", "practice": "Apply convolution filters (edge detection) to an image manually.", "time": "2.5h"},
                        {"day": "Day 100", "topic": "Pooling Layers & Strides", "theory": "CS231n: Convolutional Nets", "practice": "Understand MaxPooling calculation.", "time": "2.5h"},
                        {"day": "Day 101", "topic": "Building a CNN in Keras", "theory": "TensorFlow: CNN Tutorial", "practice": "Build a CNN for CIFAR-10 classification.", "time": "2.5h"},
                        {"day": "Day 102", "topic": "Data Augmentation", "theory": "Keras ImageDataGenerator", "practice": "Augment images (rotate, flip) to increase dataset size.", "time": "2.5h"},
                        {"day": "Day 103", "topic": "Transfer Learning (VGG16, ResNet)", "theory": "Documentation: Keras Applications", "practice": "Use a pre-trained ResNet to classify images.", "time": "2.5h"},
                        {"day": "Day 104", "topic": "CV Project", "theory": "-", "practice": "Build a Dog vs Cat Classifier.", "time": "3h"},
                        {"day": "Day 105", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 16,
                    "theme": "Sequential Data (RNNs/LSTMs)",
                    "days": [
                        {"day": "Day 106", "topic": "RNN Intuition", "theory": "Karpathy Blog: Unreasonable Effectiveness of RNNs", "practice": "Concept check: Vanishing Gradient Problem.", "time": "2.5h"},
                        {"day": "Day 107", "topic": "LSTMs (Long Short-Term Memory)", "theory": "Colah's Blog: Understanding LSTM", "practice": "Study the cell state diagrams.", "time": "2.5h"},
                        {"day": "Day 108", "topic": "GRUs (Gated Recurrent Units)", "theory": "Towards Data Science: GRU vs LSTM", "practice": "Compare GRU vs LSTM architecture.", "time": "2.5h"},
                        {"day": "Day 109", "topic": "Time Series Prediction with LSTM", "theory": "TensorFlow: Time Series", "practice": "Predict Stock Prices (simple univariate) using LSTM.", "time": "2.5h"},
                        {"day": "Day 110", "topic": "Sequence Classification (Text)", "theory": "-", "practice": "Use LSTM for Sentiment Analysis on IMDB dataset.", "time": "2.5h"},
                        {"day": "Day 111", "topic": "RNN Project", "theory": "-", "practice": "Text Generation (Character-level RNN).", "time": "3h"},
                        {"day": "Day 112", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 17,
                    "theme": "Natural Language Processing (NLP)",
                    "days": [
                        {"day": "Day 113", "topic": "NLP Basics: Text Preprocessing", "theory": "NLTK Book: Chapter 3", "practice": "Tokenization, Stemming, Lemmatization using NLTK/SpaCy.", "time": "2.5h"},
                        {"day": "Day 114", "topic": "Bag of Words & TF-IDF", "theory": "Scikit-Learn: Text Feature Extraction", "practice": "Convert text to vectors and train a classifier.", "time": "2.5h"},
                        {"day": "Day 115", "topic": "Word Embeddings (Word2Vec, GloVe)", "theory": "Jay Alammar: The Illustrated Word2Vec", "practice": "Load pre-trained GloVe vectors and find similar words.", "time": "2.5h"},
                        {"day": "Day 116", "topic": "Seq2Seq Models", "theory": "Sutskever et al. paper (concept)", "practice": "Understand Encoder-Decoder architecture.", "time": "2.5h"},
                        {"day": "Day 117", "topic": "Attention Mechanism", "theory": "Jay Alammar: Visualizing Attention", "practice": "Study the Attention logic.", "time": "2.5h"},
                        {"day": "Day 118", "topic": "NLP Project", "theory": "-", "practice": "Build a Spam Email Classifier using TF-IDF and Naive Bayes.", "time": "3h"},
                        {"day": "Day 119", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 18,
                    "theme": "Advanced NLP (Transformers)",
                    "days": [
                        {"day": "Day 120", "topic": "The Transformer Architecture", "theory": "Jay Alammar: The Illustrated Transformer", "practice": "Deep dive into Self-Attention layers.", "time": "2.5h"},
                        {"day": "Day 121", "topic": "BERT (Bidirectional Encoder)", "theory": "Google AI Blog: BERT", "practice": "Understand Masked LM and NSP.", "time": "2.5h"},
                        {"day": "Day 122", "topic": "Hugging Face Transformers Lib", "theory": "Hugging Face Course: Chapter 1", "practice": "Install transformers. Load a pipeline.", "time": "2.5h"},
                        {"day": "Day 123", "topic": "Fine-Tuning BERT", "theory": "Hugging Face Course: Fine-tuning", "practice": "Fine-tune BERT for text classification.", "time": "2.5h"},
                        {"day": "Day 124", "topic": "Variables: GPT (Generative Pre-trained)", "theory": "OpenAI: GPT-2/3 papers (Abstract)", "practice": "Generate text using GPT-2.", "time": "2.5h"},
                        {"day": "Day 125", "topic": "Transformers Project", "theory": "-", "practice": "Sentiment Analysis using DistilBERT.", "time": "3h"},
                        {"day": "Day 126", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                }
            ]
        },
        {
            "name": "Module 4: Future of Data Science + Capstone",
            "duration": "Weeks 19-28 (Months 5-7)",
            "weeks": [
                {
                    "week_num": 19,
                    "theme": "Time Series Analysis (Detailed)",
                    "days": [
                        {"day": "Day 127", "topic": "Time Series components", "theory": "Trend, Seasonality, Residuals", "practice": "Decompose a time series using statsmodels.", "time": "2.5h"},
                        {"day": "Day 128", "topic": "Stationarity & Differencing", "theory": "Augmented Dickey-Fuller Test", "practice": "Check stationarity of a dataset.", "time": "2.5h"},
                        {"day": "Day 129", "topic": "ARIMA/SARIMA Models", "theory": "Forecasting: Principles and Practice (Hyndman)", "practice": "Fit ARIMA model to forecast sales.", "time": "2.5h"},
                        {"day": "Day 130", "topic": "Facebook Prophet", "theory": "Prophet Documentation", "practice": "Install Prophet and make a quick forecast.", "time": "2.5h"},
                        {"day": "Day 131", "topic": "Machine Learning for Time Series", "theory": "XGBoost for Time Series (Sliding Window)", "practice": "Feature engineer lag features and train XGBoost.", "time": "2.5h"},
                        {"day": "Day 132", "topic": "Time Series Project", "theory": "-", "practice": "Forecast Energy Consumption.", "time": "3h"},
                        {"day": "Day 133", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                 {
                    "week_num": 20,
                    "theme": "Recommender Systems",
                    "days": [
                        {"day": "Day 134", "topic": "Recommendation Types", "theory": "Content-Based vs Collaborative Filtering", "practice": "-", "time": "2.5h"},
                        {"day": "Day 135", "topic": "Content-Based Filtering", "theory": "-", "practice": "Build a movie recommender using Metadata (Genre/Cast).", "time": "2.5h"},
                        {"day": "Day 136", "topic": "Collaborative Filtering (User-Item)", "theory": "Matrix Factorization", "practice": "Implement basic CF using cosine similarity.", "time": "2.5h"},
                        {"day": "Day 137", "topic": "Singular Value Decomposition (SVD)", "theory": "Surprise Library Docs", "practice": "Use SVD on MovieLens dataset.", "time": "2.5h"},
                        {"day": "Day 138", "topic": "Deep Learning for RecSys", "theory": "Neural Collaborative Filtering", "practice": "Research Neural CF architectures.", "time": "2.5h"},
                        {"day": "Day 139", "topic": "RecSys Project", "theory": "-", "practice": "Build a Movie/Book Recommender System.", "time": "3h"},
                        {"day": "Day 140", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 21,
                    "theme": "Generative AI & LLMs",
                    "days": [
                        {"day": "Day 141", "topic": "Generative AI Landscape", "theory": "Review: LLMs, Diffusion Models", "practice": "Browse Hugging Face Model Hub.", "time": "2.5h"},
                        {"day": "Day 142", "topic": "Prompt Engineering Basics", "theory": "learnprompting.org", "practice": "Experiment with Zero-shot, Few-shot prompting.", "time": "2.5h"},
                        {"day": "Day 143", "topic": "Chain of Thought & Advanced Prompting", "theory": "Paper: Chain-of-Thought Prompting", "practice": "Solve complex logic puzzles using CoT.", "time": "2.5h"},
                        {"day": "Day 144", "topic": "LangChain Basics", "theory": "LangChain Documentation", "practice": "Build a simple chain (Prompt + LLM).", "time": "2.5h"},
                        {"day": "Day 145", "topic": "RAG (Retrieval Augmented Generation)", "theory": "Pinecone Learning Center: RAG", "practice": "Build a simple RAG system with a text file.", "time": "2.5h"},
                        {"day": "Day 146", "topic": "GenAI Project", "theory": "-", "practice": "Create a 'Chat with your PDF' app using LangChain.", "time": "3h"},
                        {"day": "Day 147", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 22,
                    "theme": "Reinforcement Learning (RL)",
                    "days": [
                        {"day": "Day 148", "topic": "RL Basics (Agent, Environment, Reward)", "theory": "DeepMind: Intro to RL", "practice": "Understand the RL loop.", "time": "2.5h"},
                        {"day": "Day 149", "topic": "Q-Learning", "theory": "Q-Learning Explained", "practice": "Implement Q-Learning for 'FrozenLake' (Gym).", "time": "2.5h"},
                        {"day": "Day 150", "topic": "Deep Q-Networks (DQN)", "theory": "Paper: Playing Atari with Deep RL", "practice": "Review DQN architecture.", "time": "2.5h"},
                        {"day": "Day 151", "topic": "RLHF (RL from Human Feedback)", "theory": "Hugging Face Blog: RLHF", "practice": "Understand how ChatGPT was trained.", "time": "2.5h"},
                        {"day": "Day 152", "topic": "Policy Gradient Methods", "theory": "Spinning Up in Deep RL (OpenAI)", "practice": "Conceptual overview.", "time": "2.5h"},
                        {"day": "Day 153", "topic": "RL Mini-Experiment", "theory": "-", "practice": "Play with OpenAI Gym environments.", "time": "3h"},
                        {"day": "Day 154", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 23,
                    "theme": "ML Engineering & Deployment",
                    "days": [
                        {"day": "Day 155", "topic": "Pickling & Saving Models", "theory": "Joblib/Pickle", "practice": "Save your best model and load it back.", "time": "2.5h"},
                        {"day": "Day 156", "topic": "Flask/FastAPI Basics", "theory": "FastAPI Tutorial", "practice": "Create a 'Hello World' API.", "time": "2.5h"},
                        {"day": "Day 157", "topic": "Serving a Model as API", "theory": "-", "practice": "Create an endpoint that accepts JSON and returns prediction.", "time": "2.5h"},
                        {"day": "Day 158", "topic": "Docker Basics", "theory": "Docker Curriculum", "practice": "Containerize your FastAPI app.", "time": "2.5h"},
                        {"day": "Day 159", "topic": "Cloud Deployment (Intro)", "theory": "Heroku/Render/AWS Free Tier", "practice": "Deploy the Docker container to Render (Free).", "time": "2.5h"},
                        {"day": "Day 160", "topic": "Deployment Project", "theory": "-", "practice": "Deploy your Iris/Titanic model online.", "time": "3h"},
                        {"day": "Day 161", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 24,
                    "theme": "Capstone: Definition & Data",
                    "days": [
                        {"day": "Day 162", "topic": "Problem Identification", "theory": "Review: SMART Goals", "practice": "Define 3 potential Capstone ideas. Select one.", "time": "2.5h"},
                        {"day": "Day 163", "topic": "Literature Review / Market Research", "theory": "-", "practice": "Research existing solutions for your problem.", "time": "2.5h"},
                        {"day": "Day 164", "topic": "Data Collection Plan", "theory": "-", "practice": "Identify datasets. Scraping needed? APIs?", "time": "2.5h"},
                        {"day": "Day 165", "topic": "Data Acquisition", "theory": "-", "practice": "Write scripts to download/scrape data.", "time": "2.5h"},
                        {"day": "Day 166", "topic": "Initial Data Assessment", "theory": "-", "practice": "Load data, check schema, size, quality.", "time": "2.5h"},
                        {"day": "Day 167", "topic": "Capstone Milestone 1", "theory": "-", "practice": "Prepare a 1-page Proposal Document.", "time": "3h"},
                        {"day": "Day 168", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 25,
                    "theme": "Capstone: Cleaning & EDA",
                    "days": [
                        {"day": "Day 169", "topic": "Deep Data Cleaning", "theory": "-", "practice": "Handle missing values, outliers, duplicates.", "time": "2.5h"},
                        {"day": "Day 170", "topic": "Feature Engineering Strategy", "theory": "-", "practice": "Brainstorm new features relevant to the problem.", "time": "2.5h"},
                        {"day": "Day 171", "topic": "Implementing Feature Engineering", "theory": "-", "practice": "Create the features in Pandas.", "time": "2.5h"},
                        {"day": "Day 172", "topic": "Exploratory Data Analysis (Univariate)", "theory": "-", "practice": "Visualize distributions of all key variables.", "time": "2.5h"},
                        {"day": "Day 173", "topic": "Exploratory Data Analysis (Multivariate)", "theory": "-", "practice": "Explore relationships/correlations.", "time": "2.5h"},
                        {"day": "Day 174", "topic": "Capstone Milestone 2", "theory": "-", "practice": "Complete EDA Report (Notebook with comments).", "time": "3h"},
                        {"day": "Day 175", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 26,
                    "theme": "Capstone: Modeling V1",
                    "days": [
                        {"day": "Day 176", "topic": "Baseline Model", "theory": "-", "practice": "Train a simple dummy/baseline model to set benchmark.", "time": "2.5h"},
                        {"day": "Day 177", "topic": "Model Selection", "theory": "-", "practice": "Train 3-4 standard algorithms (LR, RF, XGB).", "time": "2.5h"},
                        {"day": "Day 178", "topic": "Model Evaluation", "theory": "-", "practice": "Compare models using CV and appropriate metrics.", "time": "2.5h"},
                        {"day": "Day 179", "topic": "Error Analysis", "theory": "-", "practice": "Analyze where the model is failing.", "time": "2.5h"},
                        {"day": "Day 180", "topic": "Hyperparameter Tuning", "theory": "-", "practice": "Optimize the best performing model.", "time": "2.5h"},
                        {"day": "Day 181", "topic": "Capstone Milestone 3", "theory": "-", "practice": "Finalize the best model.", "time": "3h"},
                        {"day": "Day 182", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                {
                    "week_num": 27,
                    "theme": "Capstone: Optimization & Validation",
                    "days": [
                        {"day": "Day 183", "topic": "Advanced Feature Selection", "theory": "-", "practice": "Try to simplify the model without losing performance.", "time": "2.5h"},
                        {"day": "Day 184", "topic": "Ensembling (Optional)", "theory": "-", "practice": "Combine models if performance boost is needed.", "time": "2.5h"},
                        {"day": "Day 185", "topic": "Final Evaluation on Test Set", "theory": "-", "practice": "Run final evaluation. NO MORE TUNING after this.", "time": "2.5h"},
                        {"day": "Day 186", "topic": "Interpretability", "theory": "SHAP/LIME", "practice": "Use SHAP values to explain model predictions.", "time": "2.5h"},
                        {"day": "Day 187", "topic": "Model Documentation", "theory": "-", "practice": "Document the modeling process and decisions.", "time": "2.5h"},
                        {"day": "Day 188", "topic": "Capstone Milestone 4", "theory": "-", "practice": "Code Refactoring & Cleanup.", "time": "3h"},
                        {"day": "Day 189", "topic": "Rest Day", "theory": "-", "practice": "-", "time": "0h"}
                    ]
                },
                 {
                    "week_num": 28,
                    "theme": "Capstone: Presentation & Portfolio",
                    "days": [
                        {"day": "Day 190", "topic": "Building a Demo (Streamlit)", "theory": "Streamlit Docs", "practice": "Create a simple UI for your model.", "time": "2.5h"},
                        {"day": "Day 191", "topic": "Deployment Finalization", "theory": "-", "practice": "Deploy the Streamlit app to Streamlit Cloud.", "time": "2.5h"},
                        {"day": "Day 192", "topic": "Presentation Structure", "theory": "Storytelling with Data", "practice": "Outline your slide deck.", "time": "2.5h"},
                        {"day": "Day 193", "topic": "Creating Slides", "theory": "-", "practice": "Create the presentation (Problem -> Solution -> Impact).", "time": "2.5h"},
                        {"day": "Day 194", "topic": "GitHub Profile Polish", "theory": "-", "practice": "Create a README.md, pin the project.", "time": "2.5h"},
                        {"day": "Day 195", "topic": "Mock Presentation / Video", "theory": "-", "practice": "Record a 5-min video presenting your project.", "time": "3h"},
                        {"day": "Day 196", "topic": "CELEBRATE", "theory": "-", "practice": "You are a Data Scientist! Start applying.", "time": "0h"}
                    ]
                }
            ]
        }
    ]
}
