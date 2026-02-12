
AI_ML_PLAN_DATA = {
    "title": "AI & Machine Learning Roadmap (7 Months)",
    "modules": [
        {
            "name": "Month 1: Fundamentals of AI/ML & Math",
            "duration": "Weeks 1-4",
            "weeks": [
                {
                    "week_num": 1,
                    "theme": "Overview & Environment",
                    "days": [
                        {"day": "Day 1", "topic": "AI Landscape Overview", "theory": "<b>Watch:</b> IBM Technology's <a href='https://www.youtube.com/watch?v=b4k24gR6p1E'>AI Trends for 2025 (12 mins)</a>. Then skim this <a href='https://roadmap.sh/ai-data-scientist'>visual roadmap</a> to see the big picture. Understand: <b>AI > ML > DL > Generative AI</b>.", "practice": "<b>Action:</b> Create a GitHub repo named 'AI-Learning-Journey'. Create a README.md and write a 1-paragraph commitment goal.", "time": "2h"},
                        {"day": "Day 2", "topic": "Python for ML Refresher (NumPy)", "theory": "<b>Focus:</b> NumPy is the engine of ML. Watch <a href='https://www.youtube.com/watch?v=QUT1VHiLmmI'>NumPy Crash Course (FreeCodeCamp)</a>. Fundamental concept: <b>Broadcasting</b> and <b>Vectorization</b> (avoiding loops).", "practice": "<b>Exercise:</b> Create a random 10x10 matrix. Find the mean, stdev, and transpose it without using loops.", "time": "2h"},
                        {"day": "Day 3", "topic": "Python for ML Refresher (Pandas)", "theory": "<b>Focus:</b> Data manipulation. Watch <a href='https://www.youtube.com/watch?v=vmEHCJofslg'>Pandas Tutorial (Keith Galli)</a>. Concepts: DataFrame, Series, `.loc`, `.iloc`, `.groupby()`.", "practice": "<b>Exercise:</b> Load the 'Titanic' dataset (from Seaborn or Kaggle). Fill missing age values with the median age.", "time": "2h"},
                        {"day": "Day 4", "topic": "Visualization (Matplotlib/Seaborn)", "theory": "<b>Focus:</b> Data Storytelling. Watch <a href='https://www.youtube.com/watch?v=6GUZXDef2U0'>Seaborn for Beginners</a>. Understand: Histograms (distributions) and Scatterplots (correlations).", "practice": "<b>Exercise:</b> Plot the distribution of 'Fare' in Titanic. Create a scatter plot of 'Age' vs 'Fare'.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 2,
                    "theme": "Math: Linear Algebra",
                    "days": [
                        {"day": "Day 1", "topic": "Vectors & Spaces", "theory": "<b>Visual Intuition:</b> Watch 3Blue1Brown's <a href='https://www.youtube.com/watch?v=fNk_zzaMoSs'>Vectors, what even are they?</a> and <a href='https://www.youtube.com/watch?v=k7RM-ot2NWY'>Linear combinations/span</a>. Concept: A vector is a movement in space (arrow) AND a data point (list of numbers).", "practice": "<b>Paper & Pencil:</b> Draw vectors [1, 2] and [2, 1]. Draw their sum. <b>Python:</b> Represent these as NumPy arrays and compute the sum.", "time": "2h"},
                        {"day": "Day 2", "topic": "Matrices & Operations", "theory": "<b>Visual Intuition:</b> Watch 3Blue1Brown's <a href='https://www.youtube.com/watch?v=XkY2DOUCWMU'>Matrix multiplication as composition</a>. Key concept: Matrix Mult represents transforming space (rotating, squishing).", "practice": "<b>Python:</b> Implement matrix multiplication using specific formula (3 nested loops) and compare result with `np.dot()`. Understand why `np.dot` is faster.", "time": "2h"},
                        {"day": "Day 3", "topic": "Eigenvectors & Eigenvalues", "theory": "<b>Visual Intuition:</b> Watch 3Blue1Brown's <a href='https://www.youtube.com/watch?v=PFDu9oVAE-g'>Eigenvectors and Eigenvalues</a>. Concept: The 'axis of rotation' that stays pointing in the same direction during a transformation.", "practice": "<b>Python:</b> Use `np.linalg.eig` on a simple 2x2 matrix. Verify that `A @ v = lambda * v`.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 3,
                    "theme": "Math: Calculus",
                    "days": [
                        {"day": "Day 1", "topic": "Derivatives & Rates of Change", "theory": "<b>Visual Intuition:</b> Watch 3Blue1Brown's <a href='https://www.youtube.com/watch?v=9vKqVkMQHKk'>Essence of Calculus</a>. Concept: Derivative is just the SLOPE of the graph at a specific point (how sensitive the output is to simple input changes).", "practice": "<b>Manual:</b> Calculate derivative of y = x^2 and y = 3x + 2. Relate it to 'slope'.", "time": "2h"},
                        {"day": "Day 2", "topic": "Chain Rule", "theory": "<b>Crucial for Deep Learning:</b> The Chain Rule allows us to find the error attribution in deep networks. Watch <a href='https://www.youtube.com/watch?v=YG15m2VwSjA'>Chain Rule Explained (Khan Academy)</a>.", "practice": "<b>Paper:</b> Calculate derivative of f(g(x)) where f(x) = x^2 and g(x) = sin(x).", "time": "2h"},
                        {"day": "Day 3", "topic": "Gradient Descent Intuition", "theory": "<b>The Heart of ML:</b> Watch StatQuest's <a href='https://www.youtube.com/watch?v=sDv4f4s2SB8'>Gradient Descent, Step-by-Step</a>. Concept: Taking small steps downhill to find the lowest error.", "practice": "<b>Python:</b> Implement simple gradient descent to find the minimum of y = (x-3)^2 starting from x=0.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 4,
                    "theme": "Math: Probability & Statistics",
                    "days": [
                        {"day": "Day 1", "topic": "Distributions (Normal, Binomial)", "theory": "<b>Key Concept:</b> Everything in AI is probabilistic. Watch StatQuest's <a href='https://www.youtube.com/watch?v=qBigTkBLU6g'>Normal Distribution</a>. Understand: Mean, Variance, Standard Deviation.", "practice": "<b>Python:</b> Generate 1000 random data points from a Normal Distribution using `np.random.normal`. Plot histogram.", "time": "2h"},
                        {"day": "Day 2", "topic": "Bayes Theorem", "theory": "<b>The Logic of Updating Beliefs:</b> Watch 3Blue1Brown's <a href='https://www.youtube.com/watch?v=HZGCoVF3YvM'>Bayes Theorem</a>. Formula: P(A|B) = P(B|A) * P(A) / P(B).", "practice": "<b>Problem:</b> Solve the 'Medical Test Paradox' problem on paper. (If a test is 99% accurate, do you actually have the disease?)", "time": "2h"},
                        {"day": "Day 3", "topic": "Module 1 Review", "theory": "<b>Review:</b> Re-watch 3Blue1Brown's 'Essence of Linear Algebra' Ch 1-3. These are the foundations you will use every day.", "practice": "<b>Self-Check:</b> Can you explain what a 'Dot Product' represents geometrically? (Projection/Similarity).", "time": "2h"}
                    ]
                }
            ]
        },
        {
            "name": "Month 2: Classical Machine Learning",
            "duration": "Weeks 5-8",
            "weeks": [
                {
                    "week_num": 5,
                    "theme": "Supervised Learning: Regression",
                    "days": [
                        {"day": "Day 1", "topic": "Linear Regression Concept", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=paULhF4b2hY'>Linear Regression, Clearly Explained (27 min)</a>. Concept: Fitting a line, Residuals, Least Squares.", "practice": "<b>Python:</b> Implement Simple Linear Regression using `scikit-learn` on generated dummy data `X = np.array([1,2,3...]), y = 2*X + noise`.", "time": "2h"},
                        {"day": "Day 2", "topic": "Cost Functions & Gradient Descent", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=sDv4f4s2SB8'>Gradient Descent (23 min)</a>. Concept: How the model finds the 'best' line by minimizing error.", "practice": "<b>Viz:</b> Plot the Cost Function (MSE) for different slope values `m` on your dummy data.", "time": "2h"},
                        {"day": "Day 3", "topic": "Multivariate Regression", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=zITIFTsivN8'>Multiple Linear Regression (20 min)</a>. Concept: Hyperplanes, feature weights.", "practice": "<b>Project:</b> Predict Housing Prices (Kaggle). Use `LinearRegression()` with 3+ features (e.g., Size, Rooms, Location).", "time": "2h"}
                    ]
                },
                {
                    "week_num": 6,
                    "theme": "Supervised Learning: Classification",
                    "days": [
                        {"day": "Day 1", "topic": "Logistic Regression", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=yIYKR4sgzI8'>Logistic Regression (8 min)</a>. Concept: Predicting probabilities (0-1) using the Sigmoid (Logit) function.", "practice": "<b>Python:</b> Load the Iris dataset. Use `LogisticRegression` to classify 'Setosa' vs 'Not Setosa' (Binary).", "time": "2h"},
                        {"day": "Day 2", "topic": "Metrics (Precision, Recall, F1)", "theory": "<b>Read:</b> <a href='https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9'>Accuracy, Precision, Recall or F1? (10 min)</a>. Concept: Why accuracy fails on imbalanced data.", "practice": "<b>Python:</b> Calculate Accuracy, Precision, and Recall manually for your Iris model, then verify with `sklearn.metrics`.", "time": "2h"},
                        {"day": "Day 3", "topic": "KNN & SVM", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=HVXime0nQeI'>K-Nearest Neighbors (10 min)</a> and <a href='https://www.youtube.com/watch?v=efR1C6CvhmE'>SVM (20 min)</a>.", "practice": "<b>Compare:</b> Train both KNN and Logistic Regression on the same dataset. Compare their decision boundaries (if 2D) or accuracy.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 7,
                    "theme": "Trees & Ensembles",
                    "days": [
                        {"day": "Day 1", "topic": "Decision Trees", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=7VeUPuFGJHk'>Decision Trees (18 min)</a>. Concept: Gini Impurity, information gain, root nodes.", "practice": "<b>Python:</b> Train a `DecisionTreeClassifier`. Use `plot_tree` to visualize the actual tree structure.", "time": "2h"},
                        {"day": "Day 2", "topic": "Random Forests", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=J4Wdy0Wc_xQ'>Random Forests Part 1 (14 min)</a>. Concept: Bagging (Bootstrap Aggregation) + Feature Randomness.", "practice": "<b>Python:</b> Train `RandomForestClassifier` on the 'Breast Cancer' dataset. Compare score vs single Decision Tree.", "time": "2h"},
                        {"day": "Day 3", "topic": "Project: Predict Customer Churn", "theory": "<b>Review:</b> <a href='https://www.kaggle.com/blastchar/telco-customer-churn'>Telco Customer Churn Dataset</a>. Read top notebook execution.", "practice": "<b>Action:</b> Build an end-to-end generic pipeline: cleaning -> encoding (categorical vars) -> Random Forest -> Evaluation (F1 Score).", "time": "3h"}
                    ]
                },
                {
                    "week_num": 8,
                    "theme": "Unsupervised Learning",
                    "days": [
                        {"day": "Day 1", "topic": "Clustering (K-Means)", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=4b5d3muPQmA'>K-Means Clustering (9 min)</a>. Concept: Centroids, Euclidean distance, Elbow method.", "practice": "<b>Python:</b> Use K-Means on 'Mall Customers' dataset. Plot 'Inertia' vs 'K' (Elbow Curve).", "time": "2h"},
                        {"day": "Day 2", "topic": "Dimensionality Reduction (PCA)", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=HMOI_lkzW08'>PCA Main Ideas (5 min)</a>. Concept: Projecting data onto axes of max variance.", "practice": "<b>Python:</b> Apply PCA to MNIST (784 dimensions -> 2 dimensions). Scatterplot the result colored by digit label.", "time": "2h"},
                        {"day": "Day 3", "topic": "Month 2 Capstone", "theory": "<b>Synthesize:</b> Review Regression, Classification, and Clustering. Choose one domain.", "practice": "<b>Mini-Project:</b> Take a raw dataset (e.g. 'Wine Quality'). Perform EDA, then try both RF Classification and K-Means clustering to find patterns.", "time": "3h"}
                    ]
                }
            ]
        },
        {
            "name": "Month 3: Deep Learning Basics",
            "duration": "Weeks 9-12",
            "weeks": [
                {
                    "week_num": 9,
                    "theme": "Neural Networks Foundations",
                    "days": [
                        {"day": "Day 1", "topic": "Perceptrons & Architecture", "theory": "<b>Watch:</b> 3Blue1Brown's <a href='https://www.youtube.com/watch?v=aircAruvnKk'>But what is a Neural Network? (19 min)</a>. Concept: Neurons, Layers, Weights, Biases.", "practice": "<b>Paper:</b> Draw a 2-3-1 network. Write out the formula for the output of one neuron `z = w*x + b`.", "time": "2h"},
                        {"day": "Day 2", "topic": "Activation Functions", "theory": "<b>Read:</b> <a href='https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6'>Activation Functions Explained (10 min)</a>. Understand: Why we need non-linearity (ReLU vs Sigmoid).", "practice": "<b>Python:</b> Plot `sigmoid(x)`, `tanh(x)`, and `relu(x)` using simple matplotlib.", "time": "2h"},
                        {"day": "Day 3", "topic": "Forward Pass & Backprop", "theory": "<b>Watch:</b> 3Blue1Brown's <a href='https://www.youtube.com/watch?v=Ilg3gGewQ5U'>Backpropagation Calculus (10 min)</a>. Concept: Chain rule applied to find `dLoss/dWeight`.", "practice": "<b>Conceptual:</b> Trace the path of an error signal backwards through a simple drawn network.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 10,
                    "theme": "Deep Learning Frameworks",
                    "days": [
                        {"day": "Day 1", "topic": "Intro to PyTorch/TensorFlow", "theory": "<b>Watch:</b> Daniel Bourke's <a href='https://www.youtube.com/watch?v=Z_ikDlimN6A'>PyTorch in 25 Minutes</a> (First section). Concept: Tensors are just n-dim arrays on GPU.", "practice": "<b>Python:</b> Install PyTorch. Create tensors. Perform matrix multiplication on GPU (if available) or CPU.", "time": "2h"},
                        {"day": "Day 2", "topic": "Building a NN in Code", "theory": "<b>Watch:</b> Sentdex's <a href='https://www.youtube.com/watch?v=ixwwI8G54O's>Deep Learning with PyTorch P.2 (Data)</a>. Duration: ~20 mins.", "practice": "<b>Python:</b> Define a subclass `class Net(nn.Module):`. Define 2 linear layers in `__init__`.", "time": "2h"},
                        {"day": "Day 3", "topic": "Training Loops", "theory": "<b>Read:</b> <a href='https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html'>Training a Classifier (PyTorch Blitz)</a>. Key steps: Zero grad -> Forward -> Loss -> Backward -> Step.", "practice": "<b>Python:</b> Write a training loop for your MNIST model. Train for 1 epoch.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 11,
                    "theme": "Tuning Deep Nets",
                    "days": [
                        {"day": "Day 1", "topic": "Regularization (Dropout, L2)", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=Kvx849V-5Dkk'>Regularization Part 1 (L1/L2) (20 min)</a>. Concept: Penalizing large weights.", "practice": "<b>Python:</b> Add `dropout=0.5` to your PyTorch model layers. Compare training accuracy vs test accuracy.", "time": "2h"},
                        {"day": "Day 2", "topic": "Optimizers (Adam, SGD)", "theory": "<b>Read:</b> <a href='https://ruder.io/optimizing-gradient-descent/'>An overview of gradient descent optimization algorithms</a> (Focus on Adam section).", "practice": "<b>Python:</b> Switch optimizer from `SGD` to `Adam`. Observe convergence speed.", "time": "2h"},
                        {"day": "Day 3", "topic": "Hyperparameter Tuning", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=ttE0F7fghfk'>Grid Search vs Random Search (10 min)</a>.", "practice": "<b>Python:</b> Use `GridSearchCV` (if using sklearn wrapper) or manual loop to test 3 different learning rates.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 12,
                    "theme": "Deep Learning Project",
                    "days": [
                        {"day": "Day 1", "topic": "Project Setup", "theory": "<b>Resource:</b> <a href='https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html'>CIFAR-10 Tutorial</a>.", "practice": "<b>Setup:</b> Download CIFAR-10. Create a DataLoader. Visualize 5 images.", "time": "2h"},
                        {"day": "Day 2", "topic": "Model Training", "theory": "<b>Concept:</b> Model Checkpointing. Saving the best model, not the last.", "practice": "<b>Code:</b> Implement logic to save `model.state_dict()` only if validation loss decreases.", "time": "2h"},
                        {"day": "Day 3", "topic": "Evaluation", "theory": "<b>Concept:</b> Confusion Matrix for multi-class.", "practice": "<b>Viz:</b> Create a 10x10 Heatmap of predictions vs actual labels.", "time": "2h"}
                    ]
                }
            ]
        },
        {
            "name": "Month 4: Natural Language Processing (NLP)",
            "duration": "Weeks 13-16",
            "weeks": [
                {
                    "week_num": 13,
                    "theme": "Text Basics",
                    "days": [
                        {"day": "Day 1", "topic": "Text Preprocessing", "theory": "<b>Read:</b> <a href='https://realpython.com/nltk-nlp-python/'>NLP with NLTK (Part 1)</a>. Concepts: Tokenization, Stopwords, Stemming vs Lemmatization.", "practice": "<b>Python (NLTK/Spacy):</b> Take a raw paragraph. Lowercase, remove stopwords, and tokenize it.", "time": "2h"},
                        {"day": "Day 2", "topic": "Vectorization (TF-IDF)", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=hXNbFNCgPfY'>TF-IDF (10 min)</a>. Concept: Frequency * Inverse Doc Frequency (finding 'rare' meaningful words).", "practice": "<b>Python:</b> Use `TfidfVectorizer` on a list of sentences. Inspect the resulting sparse matrix.", "time": "2h"},
                        {"day": "Day 3", "topic": "Regex & Parsing", "theory": "<b>Interactive:</b> <a href='https://regexone.com/'>RegexOne Tutorial</a> (Lessons 1-10). Pattern matching is crucial for cleaning text.", "practice": "<b>Python:</b> Write a function that extracts all email addresses and dates (dd/mm/yyyy) from a messy text string.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 14,
                    "theme": "Word Embeddings",
                    "days": [
                        {"day": "Day 1", "topic": "Word2Vec & GloVe", "theory": "<b>Read:</b> Jay Alammar's famous <a href='https://jalammar.github.io/illustrated-word2vec/'>Illustrated Word2Vec (20 min read)</a>. Concept: 'King - Man + Woman = Queen'.", "practice": "<b>Python:</b> Load pre-trained GloVe vectors (using `gensim` or `spacy`). Perform vector arithmetic (King-Man...).", "time": "2h"},
                        {"day": "Day 2", "topic": "RNNs for Text", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=AsNTP8Kwu80'>RNNs Explained (20 min)</a>. Concept: Sequential memory, unrolling across time.", "practice": "<b>Python:</b> Build a simple `nn.RNN` layer in PyTorch. Pass a fake sentence tensor through it.", "time": "2h"},
                        {"day": "Day 3", "topic": "LSTMs/GRUs", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=YCzL96nL7j0'>LSTMs Explained (26 min)</a>. Concept: Forget gate, input gate, overcoming vanishing gradients.", "practice": "<b>Python:</b> Replace your RNN layer with `nn.LSTM`. Output the final hidden state.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 15,
                    "theme": "Modern NLP (Transformers)",
                    "days": [
                        {"day": "Day 1", "topic": "Attention Mechanism", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=PSs6nxngL6k'>Attention (20 min)</a> OR Read <a href='https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/'>Visualizing Neural Machine Translation</a>.", "practice": "<b>Conceptual:</b> Draw the Attention matrix. What does it tell us about word relationships?", "time": "2h"},
                        {"day": "Day 2", "topic": "BERT & GPT Concepts", "theory": "<b>Read:</b> Jay Alammar's <a href='https://jalammar.github.io/illustrated-bert/'>Illustrated BERT (15 min read)</a>. Concept: Masked Language Modeling vs Next Token Prediction.", "practice": "<b>Examine:</b> Look at the architecture diagrams. Note the difference between Encoder (BERT) and Decoder (GPT).", "time": "2h"},
                        {"day": "Day 3", "topic": "Hugging Face Library", "theory": "<b>Watch:</b> Hugging Face's <a href='https://www.youtube.com/watch?v=tiZFewofSLM'>Transformers Library Tour (10 min)</a>. Concepts: `from_pretrained`, Pipelines.", "practice": "<b>Python:</b> Use `pipeline('sentiment-analysis')` to classify 5 sentences with zero training.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 16,
                    "theme": "NLP Project",
                    "days": [
                        {"day": "Day 1", "topic": "Project: Text Classification", "theory": "-", "practice": "Fine-tune a model for IMDb reviews.", "time": "2h"},
                        {"day": "Day 2", "topic": "Pipeline Integration", "theory": "-", "practice": "Build a clean inference function.", "time": "2h"},
                        {"day": "Day 3", "topic": "Review", "theory": "-", "practice": "Compare TF-IDF vs BERT performance.", "time": "2h"}
                    ]
                }
            ]
        },
        {
            "name": "Month 5: Computer Vision",
            "duration": "Weeks 17-20",
            "weeks": [
                {
                    "week_num": 17,
                    "theme": "CNN Fundamentals",
                    "days": [
                        {"day": "Day 1", "topic": "Convolutions & Pooling", "theory": "<b>Watch:</b> StatQuest's <a href='https://www.youtube.com/watch?v=HGwBXDKFk9I'>Convolutional Neural Networks (15 min)</a>. Concept: Filters detecting edges, pooling reducing size.", "practice": "<b>Python:</b> Apply a vertical edge detection filter `[[1,0,-1], [1,0,-1], [1,0,-1]]` to a grayscale image manually (using scipy/numpy).", "time": "2h"},
                        {"day": "Day 2", "topic": "CNN Architecture", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=YRhxdVk_sIs'>DeepLearning.AI: Simple CNN Network (10 min)</a>. Structure: Conv -> ReLU -> Pool -> FC.", "practice": "<b>Python:</b> Build a standard CNN in PyTorch (`nn.Conv2d`, `nn.MaxPool2d`).", "time": "2h"},
                        {"day": "Day 3", "topic": "Training CNNs", "theory": "<b>Concept:</b> Data Augmentation importance for images.", "practice": "<b>Python:</b> Train your CNN on FashionMNIST. Use `torchvision.transforms` to add rotation/flips.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 18,
                    "theme": "Advanced Architectures",
                    "days": [
                        {"day": "Day 1", "topic": "Classic Architectures", "theory": "<b>Read:</b> <a href='https://medium.com/platform-engineer/cnn-architecture-series-from-lenet-5-to-efficientnet-part-1-3063f2e11832'>CNN Architectures: LeNet to ResNet (10 min)</a>.", "practice": "<b>Python:</b> Load `torchvision.models.resnet18(pretrained=True)`. Print the model summary.", "time": "2h"},
                        {"day": "Day 2", "topic": "Transfer Learning", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=K-f23RK3Gxs'>Transfer Learning Explained (10 min)</a>. Concept: Freezing early layers, training only the head.", "practice": "<b>Python:</b> Fine-tune ResNet18 to classify 'Ants vs Bees' (download small dataset).", "time": "2h"},
                        {"day": "Day 3", "topic": "Data Augmentation", "theory": "<b>Viz:</b> Look at how augmentation changes images. Read PyTorch docs on Transforms.", "practice": "<b>Code:</b> Create a visualization grid showing 1 original image and 5 augmented versions of it.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 19,
                    "theme": "CV Tasks",
                    "days": [
                        {"day": "Day 1", "topic": "Object Detection Intro", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=9ahXVcIomTE'>YOLO Algorithm Explained (15 min)</a>. Concept: Grid Division + Bounding Boxes.", "practice": "<b>Python (Inference):</b> Load `ultralytics/yolov5` from Hub. Run inference on a photo found on Google Images.", "time": "2h"},
                        {"day": "Day 2", "topic": "Image Segmentation Intro", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=azM57NuW0K8'>U-Net Architecture Explained (10 min)</a>. Concept: Pixel-wise classification.", "practice": "<b>Demo:</b> Run a pre-trained segmentation model (Mask R-CNN) from torchvision on an image.", "time": "2h"},
                        {"day": "Day 3", "topic": "CV Project", "theory": "<b>Setup:</b> Build a webcam Face Detector.", "practice": "<b>Code:</b> Use OpenCV `cv2.CascadeClassifier` (Haar Cascades) or a DL model for real-time face detection.", "time": "3h"}
                    ]
                },
                {
                    "week_num": 20,
                    "theme": "Month 5 Review",
                    "days": [
                        {"day": "Day 1", "topic": "Review", "theory": "Consolidate CNN knowledge.", "practice": "Refactor your CV project code.", "time": "2h"}
                    ]
                }
            ]
        },
        {
            "name": "Month 6: Generative AI & RAG",
            "duration": "Weeks 21-24",
            "weeks": [
                {
                    "week_num": 21,
                    "theme": "LLM Fundamentals",
                    "days": [
                        {"day": "Day 1", "topic": "LLM Architecture", "theory": "<b>Watch:</b> Andrej Karpathy's <a href='https://www.youtube.com/watch?v=zjkBMFhNj_g'>Intro to Large Language Models (1 hour)</a>. (You can split this over 2 days if needed, focus on the first 30 mins: Concept of Next Token Prediction).", "practice": "<b>Play:</b> Experiment with <a href='https://platform.openai.com/playground'>OpenAI Playground</a> (or HuggingChat). Try changing temperature and system prompts.", "time": "2h"},
                        {"day": "Day 2", "topic": "Prompt Engineering", "theory": "<b>Course:</b> <a href='https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/'>DeepLearning.AI: Prompt Engineering for Developers</a> (Watch first 3 chapters - 20 mins).", "practice": "<b>Code:</b> Write a Python script using `openai` lib to perform 'Few-Shot' classification on movie reviews.", "time": "2h"},
                        {"day": "Day 3", "topic": "API Integration", "theory": "<b>Read:</b> <a href='https://platform.openai.com/docs/quickstart'>OpenAI API Quickstart</a>.", "practice": "<b>Code:</b> Build a CLI Chatbot. Allow the user to type, send to GPT-3.5/4, and print the response. Keep chat history in a list.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 22,
                    "theme": "RAG (Retrieval Augmented Generation)",
                    "days": [
                        {"day": "Day 1", "topic": "RAG Concepts", "theory": "<b>Watch:</b> IBM Technology's <a href='https://www.youtube.com/watch?v=T-D1OfcDW1M'>What is RAG? (10 min)</a>. Concept: Retrieving relevant context to reduce hallucinations.", "practice": "<b>Conceptual:</b> Draw the RAG pipeline: User Query -> Embed -> Search Vector DB -> Retrieve -> Prompt + Context -> LLM.", "time": "2h"},
                        {"day": "Day 2", "topic": "Vector Databases", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=dN0lsF2cvm4'>Vector Databases Explained (15 min)</a>. Concept: Cosine Similarity.", "practice": "<b>Python:</b> Use `chromadb` or `faiss`. Store 5 sentences. Query for the most similar one to 'I like food'.", "time": "2h"},
                        {"day": "Day 3", "topic": "Building a RAG Pipeline", "theory": "<b>Watch:</b> LangChain's <a href='https://www.youtube.com/watch?v=LhnCsygAVzY'>RAG from Scratch (20 min)</a>.", "practice": "<b>Code:</b> Create a script that reads a text file (e.g., your resume), chunks it, and answers questions about it using RAG.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 23,
                    "theme": "Frameworks (LangChain)",
                    "days": [
                        {"day": "Day 1", "topic": "LangChain/LlamaIndex", "theory": "<b>Watch:</b> Rabbitmetrics' <a href='https://www.youtube.com/watch?v=aywZRZzwSqY'>LangChain Explained in 13 Minutes</a>. Concept: Chains and Agents.", "practice": "<b>Code:</b> Build a 'SequentialChain' that takes a topic, generates a title, and then writes a poem about that title.", "time": "2h"},
                        {"day": "Day 2", "topic": "Evaluation", "theory": "<b>Read:</b> <a href='https://docs.ragas.io/en/latest/concepts/metrics/fidelity.html'>RAGAS Metrics</a>. Concept: Faithfulness and Answer Relevance.", "practice": "<b>Manual Eval:</b> Ask your RAG 10 control questions. Manually rate the answers 1-5.", "time": "2h"},
                        {"day": "Day 3", "topic": "GenAI Project", "theory": "<b>Setup:</b> Streamlit + LangChain.", "practice": "<b>Project:</b> Build 'ChatWithPDF'. User uploads PDF, system answers questions. (Search 'ChatWithPDF LangChain tutorial' on YT, aim for < 30 min video).", "time": "3h"}
                    ]
                },
                {
                    "week_num": 24,
                    "theme": "Month 6 Review",
                    "days": [
                         {"day": "Day 1", "topic": "Review", "theory": "Consolidate GenAI concepts.", "practice": "Polish your AI Assistant.", "time": "2h"}
                    ]
                }
            ]
        },
        {
            "name": "Month 7: Final Project & Portfolio",
            "duration": "Weeks 25-28",
            "weeks": [
                {
                    "week_num": 25,
                    "theme": "Project Ideation & Design",
                    "days": [
                        {"day": "Day 1", "topic": "Problem Selection", "theory": "<b>Read:</b> <a href='https://buffer.com/resources/product-specs/'>How to write a Product Spec</a>. Don't start coding yet.", "practice": "<b>Doc:</b> Write a 1-page Spec. Problem, Solution, Tech Stack, User Stories.", "time": "2h"},
                        {"day": "Day 2", "topic": "Architecture Design", "theory": "<b>Viz:</b> <a href='https://excalidraw.com/'>Excalidraw</a>. Draw how your Frontend connects to Backend and Model.", "practice": "<b>Diagram:</b> Create the system architecture diagram. Save it to your repo.", "time": "2h"},
                        {"day": "Day 3", "topic": "Stack Setup", "theory": "<b>Tool:</b> <a href='https://streamlit.io/'>Streamlit</a> vs <a href='https://fastapi.tiangolo.com/'>FastAPI</a>. Choose Streamlit for speed.", "practice": "<b>Code:</b> Initialize git repo. Set up `requirements.txt`. Create a 'Hello World' Streamlit app.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 26,
                    "theme": "Implementation",
                    "days": [
                        {"day": "Day 1", "topic": "Core Logic / Modeling", "theory": "<b>Focus:</b> The AI Brain.", "practice": "<b>Code:</b> Implement the core python class that handles the Logic (e.g., `class PDFChatbot`). Test it in a notebook without UI.", "time": "3h"},
                        {"day": "Day 2", "topic": "Backend API", "theory": "<b>Focus:</b> Connectivity.", "practice": "<b>Code:</b> If using Streamlit, integrate your Class into `app.py`. If FastAPI, create an endpoint `/predict`.", "time": "3h"},
                        {"day": "Day 3", "topic": "Frontend UI", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=D0D4Pa22iG0'>Streamlit Crash Course (20 mins)</a>.", "practice": "<b>Code:</b> Add file uploader, chat input, and history display to your app.", "time": "3h"}
                    ]
                },
                {
                    "week_num": 27,
                    "theme": "Refinement & Theory",
                    "days": [
                        {"day": "Day 1", "topic": "Gap Filling", "theory": "<b>Review:</b> Check your backlog of concepts.", "practice": "<b>Fix:</b> Fix known bugs in your project. Add error handling (try/except).", "time": "2h"},
                        {"day": "Day 2", "topic": "Optimization", "theory": "<b>Concept:</b> Latency.", "practice": "<b>Code:</b> Add a loading spinner. Cache expensive function calls using `@st.cache_resource`.", "time": "2h"},
                        {"day": "Day 3", "topic": "Deployment", "theory": "<b>Guide:</b> <a href='https://huggingface.co/docs/hub/spaces-sdks-streamlit'>Deploy Streamlit to Hugging Face Spaces</a>.", "practice": "<b>Action:</b> Push code to HF Spaces. Verify it works publicly.", "time": "2h"}
                    ]
                },
                {
                    "week_num": 28,
                    "theme": "Portfolio & Completion",
                    "days": [
                        {"day": "Day 1", "topic": "Documentation", "theory": "<b>Read:</b> <a href='https://github.com/matiassingers/awesome-readme'>Awesome README</a>. A good README is the difference between a project being used or ignored.", "practice": "<b>Doc:</b> Write a README.md for your Capstone. Include: Demo GIF, Tech Stack, Setup Instructions.", "time": "2h"},
                        {"day": "Day 2", "topic": "Showcase", "theory": "<b>Watch:</b> <a href='https://www.youtube.com/watch?v=S5C_z1nVD_o'>How to Demo Software (10 min)</a>.", "practice": "<b>Video:</b> Record a 2-minute Loom video walking through your AI app. Post it on LinkedIn/X.", "time": "2h"},
                        {"day": "Day 3", "topic": "Final Audit & Next Steps", "theory": "<b>Read:</b> <a href='https://roadmap.sh/ai-data-scientist'>The Roadmap (Revisited)</a>. See how far you've come.", "practice": "<b>Career:</b> Update your Resume to include: 'Built End-to-End RAG Chatbot using LangChain & Streamlit'. Apply to 1 job.", "time": "2h"}
                    ]
                }
            ]
        }
    ]
}
