# Portfolio
---
Hello there! I'm Clement and this is a summary of the projects I have worked on and I have included their respective GitHub links for more technical details. If you're interested to know more, feel free to [get in touch](mailto:clementow.zs@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/clement-ow/))!


#### Contents
- [Data Science](https://github.com/thisisclement/portfolio#data-science)

---
## Data Science

### 1. SAT and ACT scores analysis
Oct 2019 | https://github.com/thisisclement/SAT-ACT-test-score-analysis

<p align="center">
<img width="500" src="./images/proj_1.png"  />
</p>

##### Problem Statement
There are many States with their declining SAT or ACT participation rates or worse, not even moved in between years. This paper proves to see what the trends are in the SAT and ACT scores and participation rates and digs deep using statistics to find out why some of it might be the case.  Finally, conclusions were drawn and recommendations to increase SAT participation rates for a chosen state were proposed.

##### Summary of Findings
High participation rates for one test usually means low participation rates in the other, a trend that is especially true for states where one of the tests is mandatory. As such, efforts to increase participation rates for the SAT should be diverted away from states currently with mandatory ACT testings, as they may not be as effective in these states.

Based on the data examined, the data shows that by making a particular test mandatory or by offering free administration of the test, it increases the probability of the participation rate to be higher. The highest 6 States with a 100% participation rate on either years all are the result of making them mandatory or offered free.

The chosen state to increase participation rate is Iowa as it is one of the bottom two states (together with South Dakota) with the lowest participation rates for SAT tests. It seems to be the better choice with a much higher population which in turn benefits more high school students to get a chance to go to University and a relatively median household income very close to the national median to support potential tuition fees. Furthermore, more Universities are implementing a hybrid scoring approach taking into account their socio-economic status in conjunction with SAT scores for fairer University admissions.


##### The Process
- data cleaning
- exploratory data analysis of trends in scores and participation rates
- make and justify recommendations
- presentation of findings

##### Language
Python

##### Key Libraries
`pandas`, `NumPy`, `matplotlib`, `seaborn`


### 2. Ames Housing Price Prediction
Oct 2019 | https://github.com/thisisclement/Ames-Housing-Price-Prediction

<p align="center">
  <img width="700" src="./visualisations/project2.png"  />
</p>

##### Problem Statement
A comprehensive housing dataset from the city of Ames in Iowa, USA ([source](https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge/overview)) was examined. Predictors of housing prices can vary from state to state. With 70 over possible predictors, it is essential to analyse the top five predictors that can highly influence housing prices. This information can help inform buyers to look for the right kind of feature in a house or even help home owners and property companies better price them.

##### Summary of Findings
Tested 4 different regression models with various feature engineering and selection techniques. Ridge Regression came up as the best model with an R<sub>2</sub> score of `88.82%` and does not look to be overfitted as they perform better on the test score.

Apart from the area and age of the property, the top feature that highly influences the price is a property located at Stone Brooks neighbourhood. It is no wonder the best predictor as this neighbourhood is very convenient and is the closest to Iowa State University, the largest university in Iowa state. It is also located very close to downtown. The other two neighbourhoods, Northridge Heights and Northridge are situated close together and is slightly further away to then University but is equidistant to downtown as compared to Stone Brook neighbourhood. Moreover all three neighbourhoods are very close to elementary, middle and high schools and have amenities really close by.

However, it is interesting to note that the land contour that is hilly is valued more than the other land contour features. It is perhaps has a good privacy rating among home dwellers in Ames.
At number five, Overall Quality of the property takes the spot and is no surprise that it is one of the top predictors in property prices, though the coefficient is much lower than being in the top three neighbourhoods.

It is important to note that the above conclusions are only based on the Ames housing dataset from 2006 to 2010. In order to complement the above findings, more granular data such as buyer data and coordinates of home sales would be beneficial to delve deep into analysis such as buyer behaviour or neighbourhood studies. Other factors not included in the dataset like economic conditions, buyer mindset and psychology which can influence the buyers and sellers.  

##### The Process
- deal with missing data, outliers, and skewed distributions
- engineer new features such as age sold
- model tuning and evaluations
- feature selection and identification of final model
- make and justify recommendations
- presentation of findings

##### Language
Python

##### Key Libraries
`scikit-learn`: `LinearRegression`, `Lasso`, `Ridge`, `ElasticNet`, `model_selection`


### 3. Subreddit Classification
Oct 2019 |

<p align="center">
  <img width="700" src="./visualisations/project3.png"  />
</p>

##### Problem Statement


##### Summary of Findings


##### The Process


##### Language
Python

##### Key Libraries
`requests`, `PRAW`, `regex`, `spacy`, `nltk`, `scikit-learn`: `CountVectorizer`, `TfidfVectorizer`, `Pipeline`, `LogisticRegression`, `KNeighborsClassifier`, `MultinomialNB`


### 4. West Nile Virus Prediction in Chicago
Nov 2019 |

<p align="center">
  <img width="500" src="./visualisations/project4.png"  />
</p>

##### Problem Statement

##### Summary of Findings

##### The Process


##### Language
Python

##### Key Libraries


### 5. Online Hate Speech Prediction
Nov 2019 | https://github.com/thisisclement/Hate-Speech-Prediction

<p align="center">
  <img width="700" src="./visualisations/capstone.gif"  />
</p>

##### Problem Statement


##### Summary of Findings

##### The Process


##### Language & Tools


##### Key Libraries
`requests`, `scikit-learn`, `keras`, `Flask`
