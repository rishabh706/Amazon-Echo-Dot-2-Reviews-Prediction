# Natural Language Processing
#setwd('D:/Data Science Project/Amazon echo Dot 2 reviews prediction')
# Importing the dataset
dataset=read.csv("Amazon Echo 2 Reviews.csv",stringsAsFactors = FALSE)
dataset=dataset[c('Title','Review.Text','Rating')]
dataset$Rating=ifelse(dataset$Rating>3,1,0)
dataset$Rating=as.integer(dataset$Rating)
sapply(dataset,typeof)
apply(dataset,2,function(col)sum(is.na(col))/length(col))
summary(dataset)
# Cleaning the texts
#install.packages('NLP')
library(NLP)
library(tm)
library(SnowballC)
corpus=VCorpus(VectorSource(dataset[c(-3)]))
corpus=tm_map(corpus,content_transformer(tolower))
corpus=tm_map(corpus,removeNumbers)
corpus=tm_map(corpus,removePunctuation)
corpus=tm_map(corpus,removeWords,stopwords())
corpus=tm_map(corpus,stemDocument)
corpus=tm_map(corpus,stripWhitespace)

# Creating the Bag of words Model
dtn=DocumentTermMatrix(corpus)
dtn=removeSparseTerms(dtn,0.999)

# Encoding the Categorical Data
dataset$Rating=factor(dataset$Rating,levels = c(0,1))

# Splitting the dataset into the training set and the test set
library(caTools)
set.seed(123)
split=sample.split(dataset$Rating,SplitRatio = 0.80)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

# Fitting the Logistic Regression to the Training set
classifier=glm(formula=Rating ~.,family=binomial,data=training_set)

# Predicting the test set results
y_pred=predict(classifier,newdata=test_set[-3])









