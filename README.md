\# Movie Rating Prediction with Python



\## 📌 Project Overview



This project focuses on predicting the IMDb rating of Indian movies using Machine Learning and Python.



The model uses movie-related information such as \*\*Genre, Director, Actors, Year, and Duration\*\* to predict the IMDb rating of a movie.



The project follows a complete Data Science workflow:



\* Data loading

\* Data cleaning

\* Exploratory Data Analysis (EDA)

\* Feature engineering

\* Model training

\* Model evaluation

\* Model comparison

\* Prediction



\---



\## 📊 Dataset



The project uses the \*\*IMDb Movies India\*\* dataset.



The original dataset contains \*\*15,509 movies\*\* and 10 columns.



Important columns include:



\* `Name` — Movie name

\* `Year` — Release year

\* `Duration` — Movie duration

\* `Genre` — Movie genre

\* `Rating` — IMDb rating

\* `Votes` — Number of votes

\* `Director` — Movie director

\* `Actor 1` — First actor

\* `Actor 2` — Second actor

\* `Actor 3` — Third actor



After cleaning the data, movies without an IMDb rating were removed.



\* Original movies: \*\*15,509\*\*

\* Movies without rating removed: \*\*7,590\*\*

\* Movies used for modeling: \*\*7,919\*\*



\---



\## 🧹 Data Cleaning



The following preprocessing steps were performed:



\* Extracted the numerical year from the `Year` column

\* Extracted movie duration in minutes

\* Converted votes into numerical values

\* Removed movies without ratings

\* Filled missing categorical values with `"Unknown"`

\* Prepared the dataset for machine learning



\---



\## 🔎 Exploratory Data Analysis



Several visualizations were created to understand the dataset and rating patterns.



The following charts are generated automatically:



1\. Rating Distribution

2\. Rating by Genre

3\. Rating by Year

4\. Correlation Heatmap



The charts are saved inside the `outputs/` directory.



\---



\## ⚙️ Feature Engineering



The project uses several engineered features, including:



\* Genre multi-hot encoding

\* Director mean encoding

\* Actor mean encoding

\* Director movie count

\* Year

\* Duration



Mean encoding was performed using out-of-fold information to reduce data leakage.



The `Votes` feature is disabled by default in the project.



\---



\## 🤖 Machine Learning Models



Three regression models were trained and compared:



\### 1. Ridge Regression



A regularized linear regression model used as a machine learning baseline.



\### 2. Random Forest Regressor



An ensemble tree-based regression model capable of learning nonlinear relationships.



\### 3. Gradient Boosting Regressor



An ensemble model that builds multiple decision trees sequentially to improve prediction performance.



A simple mean-rating baseline was also used for comparison.



\---



\## 📈 Model Performance



The models were evaluated using:



\* MAE — Mean Absolute Error

\* RMSE — Root Mean Squared Error

\* R² — R-squared



| Model                 |       MAE |      RMSE |        R² |

| --------------------- | --------: | --------: | --------: |

| Baseline (mean)       |     1.107 |     1.365 |    -0.002 |

| Ridge Regression      |     0.880 |     1.133 |     0.310 |

| Random Forest         |     0.865 |     1.118 |     0.327 |

| \*\*Gradient Boosting\*\* | \*\*0.853\*\* | \*\*1.099\*\* | \*\*0.350\*\* |



\### 🏆 Best Model



The \*\*Gradient Boosting Regressor\*\* achieved the best performance based on RMSE.



\* \*\*MAE:\*\* 0.853

\* \*\*RMSE:\*\* 1.099

\* \*\*R²:\*\* 0.350



\---



\## 🔥 Important Features



The top features identified by the model were:



1\. `Actor\\\_mean`

2\. `Director\\\_enc`

3\. `Year`

4\. `Actor1\\\_enc`

5\. `Duration`

6\. `Actor2\\\_enc`

7\. `Actor3\\\_enc`

8\. `Director\\\_movies`

9\. `Genre\\\_Documentary`

10\. `Genre\\\_Horror`



`Actor\\\_mean` was the most important feature in the final model.



\---



\## 🎬 Sample Predictions



The trained model was used to predict ratings for movies in the test dataset.



Example predictions include:



| Movie                     | Actual Rating | Predicted Rating |

| ------------------------- | ------------: | ---------------: |

| Mohabbat Pehli Nazar Mein |           3.3 |              4.9 |

| Veerappan                 |           5.3 |              5.0 |

| Commando 3                |           5.7 |              4.9 |

| Dhoop                     |           7.2 |              5.8 |

| Horror Night              |           3.5 |              5.4 |

| Saheb Biwi Aur Gangster   |           7.2 |              6.1 |

| Sameera                   |           3.8 |              6.0 |

| Fashion                   |           6.9 |              5.0 |

| Main Suhagan Hoon         |           5.2 |              5.7 |

| Firaaq                    |           7.4 |              6.8 |



\---



\## 📁 Project Structure



```text

Movie Rating Prediction/

│

├── IMDb Movies India.csv

├── movie\\\_rating\\\_prediction.py

├── README.md

│

└── outputs/

\&#x20;   ├── 01\\\_rating\\\_distribution.png

\&#x20;   ├── 02\\\_rating\\\_by\\\_genre.png

\&#x20;   ├── 03\\\_rating\\\_by\\\_year.png

\&#x20;   ├── 04\\\_correlation\\\_heatmap.png

\&#x20;   ├── 05\\\_model\\\_comparison.png

\&#x20;   ├── 06\\\_actual\\\_vs\\\_predicted.png

\&#x20;   ├── 07\\\_feature\\\_importance.png

\&#x20;   └── model\\\_comparison.csv

```



\---



\## 🛠️ Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Matplotlib

\* Seaborn



\---



\## 🚀 How to Run



\### 1. Install the required libraries



```bash

pip install pandas numpy scikit-learn matplotlib seaborn

```



\### 2. Keep the dataset in the same folder as the Python script



The expected dataset filename is:



```text

IMDb Movies India.csv

```



\### 3. Run the Python script



```bash

python movie\\\_rating\\\_prediction.py

```



The program will perform data cleaning, EDA, feature engineering, model training, evaluation, and prediction automatically.



All generated charts and model comparison results will be saved in the `outputs/` folder.



\---



\## 📌 Conclusion



This project demonstrates how Machine Learning can be used to predict movie ratings using information about Indian movies.



Among the tested models, \*\*Gradient Boosting Regressor performed the best\*\*, achieving an RMSE of \*\*1.099\*\* and an R² score of \*\*0.350\*\* on the test dataset.



The project provided practical experience with data preprocessing, exploratory data analysis, feature engineering, regression models, model evaluation, and prediction.



\---







\*\*Debarpan Ganguly\*\*



Data Science \& Machine Learning Project

