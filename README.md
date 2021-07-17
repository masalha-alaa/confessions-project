# Confessions Project

### Table of Contents
[Introduction](#introduction)

[Classification (Machine Learning)](#ml-classif)

[Classification (Deep Learning)](#dl-classif)

[Text Generation (RNN, Deep Learning)](#text-gener)

---

<a name="introduction"/>

## Introduction

In this project I used [facebook-scraper](https://github.com/masalha-alaa/facebook-scraper) to fetch 12000 Facebook MIT Confession posts, and tried to do various things with them.
My initial goal was Author Clustering. Namely, to map posts that belong to the same author together. Since the confessions are anonymous (no labels available), the only possible approach would be unsupervised learning. Unfortuntaely, I read a few papers that attempted to tackle this problem (authership clustering), such as [Clustering by Authorship Within and Across Documents, by Efstathios Stamatatos Et Al.](http://ceur-ws.org/Vol-1609/16090691.pdf), and the results were not promising. Other papers that use a supervised method, however, present promising results. For example: [\[1\]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf) and [\[2\]](https://arxiv.org/pdf/1912.10204.pdf). But since I can't use a supervised method with my dataset, I ditched this idea altogether.

My next thought was to perform some type of classification. My dataset included: Post ID, Date, Text, Number of Likes. So I decided to try and predict the number of likes based on some features. Unfortunately, the likes distribution was very skewed:

![likes density distribution](https://user-images.githubusercontent.com/78589884/125991986-070e2821-d8d5-43c3-a6dd-0c66a7cfde03.png)

Maybe I could have deskewed the distribution by applying log on the target, and then predict the number of likes using a Regressor. While that might have been a nice approach, I decided to make my life easier and go on with another direction.

<a name="ml-classif"/>

## Classification (Machine Learning)
I decided to simply predict whether the post received at least 9 likes or not. I chose the number 9 because it divides the dataset to 50-50 so it's balanced.

To make it more interesting, I decided to try 2 approaches: a regular machine learning classifier, and an artificial neural network (deep learning).

But first, I had to think of some features for the task. Here's what I came up with:
* 20 most common words in the dataset.
* 20 most common word bigrams in the dataset.
* 20 most common character trigrams in the dataset.
* 20 best word features from the dataset according to Scikit-Learn's [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).
* Sentiment analysis (a classifier that rates each post as positive / negative / neutral / compound).
* The year of the post.
* The month of the post.
* The weekday of the post.
* The length of the post (in words).
* The average, min, and max lengths of the words in the post.
* Number of exclamation marks in the post.

I collected all these features, vectorized and scaled<sup>*</sup> them (tf-idf values for the first 5, and normal scaling for the others).

Next, I tried several classifiers, such as: SVM, Naive Bayes (NB), Decision Tree (DT), Random Forest (RF), and KNN (code under [/prediction/main.py](https://github.com/masalha-alaa/confessions-project/blob/master/prediction/main.py)). The best three were: RF > DT > NB. The RF's result was 72% on the training set, and 63% on the test set. Given that our random chance baseline is 50%, it seems that the classifier has learned something; but I was really disappointed, since I didn't think this was such a difficult task. I looked at the classifier's features ranking, and here are its top 10 in a decreasing order:

|   Rank        | Feature              | Importance    |
| ---           | ---                  | ---           |
| 1.            | Year Timestamp       | 0.177         |
| 2.            | Month Timestamp      | 0.053         |
| 3.            | Post Length          | 0.048         |
| 4.            | Distinct Words       | 0.045         |
| 5.            | Hour Timestamp       | 0.040         |
| 6.            | Average Word Length  | 0.040         |
| 7.            | mit (word)           | 0.032         |
| 8.            | compound (sentiment) | 0.032         |
| 9.            | neutral (sentiment)  | 0.029         |
| 10.           | positive (sentiment) | 0.027         |

<sup>*</sup> _The scaling is not necessary for RF and DT, but it's beneficial for the others._

It's interesting how the year timestamp is the most dominant feature. Here's how its distribution looks like in a graph:

![year timestamp graph](https://user-images.githubusercontent.com/78589884/125993860-aab12ffd-9100-4e7f-89fb-c8addadf27e6.png)  
_the y-axis is the popularity of the post (it's "popular" if it received >= 9 likes)_

Here's how other features are distributed among the dataset:
![top features distribution](https://user-images.githubusercontent.com/78589884/125994199-e24e60e0-5c11-425c-a57b-7a7bad1b9c46.png)

<a name="dl-classif"/>

## Classification (Deep Learning)

Not being satisfied with the RF results, I decided to give the RNN network a try. So I built a simple LSTM model, and fed it with the same features as I did for the RF. Unfortunately, the results were disappointing. The LSTM network didn't achieve more than a 57% accuracy after 30 epochs:

![RNN plot](https://user-images.githubusercontent.com/78589884/125994843-9ee357a5-13e0-4725-bb2c-e0d5e951775d.png)

So it almost didn't learn anything.

The code for the deep learning approach can be fount at [/prediction/confessions_prediction_dl.ipynb](https://github.com/masalha-alaa/confessions-project/blob/master/prediction/confessions_prediction_dl.ipynb).

<a name="text-gener"/>

## Text Generation (RNN, Deep Learning)
Not being satisfied with the results, I decided to take this one step further. I wanted to build a generative AI model, that generates confessions from thin air, after being trained on my confessions dataset. I found this [TensorFlow tutorial](https://www.tensorflow.org/text/tutorials/text_generation), in which they built such a model to generate Shakespeare writings. With a few modifications, I managed to adapt it to my own needs and make it work with my confessions dataset. The results are real fun to look at:

![confessions-generation-gif](https://user-images.githubusercontent.com/78589884/125989278-ba093243-f2df-4852-9feb-bfcb803d598a.gif)

As you can see, it's nowhere near perfect, but it's quite impressive how it quickly learned the structure of the posts (Date, Confession Number, Text). At the beginning it writes out straight gibberish, but then its English starts improving little by little, until it converges after about 20 epochs.

Some of the confessions it generated:

_2018-03-20 07:38:00_  
_#12987_  
_I reached the other day, every time I see in a big crush on "1 "social" but he's gonna drive_

_2021-05-28 10:22:10_  
_#7996_  
_Am I the only thing that I used to my Internship and my world is leftist_

_2015-12-17 02:53:53_  
_#3385_  
_Honestly, math compenses should doesn't make you better. So I literally don't belong and confused."_

Obviously those don't make a lot of sense, but I hope you at least had a laugh!

Next I'm going to deploy the model on some hosting website, and let it generate new confessions on a button click so people can have some fun!
