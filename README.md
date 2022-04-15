# Language recognition neural network
## Table of contents
* [Introduction](#introduction)
* [Data description](#data-description)
* [Examples of performance](#examples-of-performance)
* [Methods used](#methods-used)

## Introduction
A single-layer neural network written from scratch that predicts the language of the text based on the proportions of letters.

## Data description
Articles scraped from Wikipedia. The neural network was trained on 3 languages (English, German, Polish), 10 articles each. However, the code is generic and can be applied to any number of languages. 

## Examples of performance
English sample:

![image](https://user-images.githubusercontent.com/74184204/162751513-e08b074d-5195-432e-954b-3f6f9509c94e.png)

German sample:

![image](https://user-images.githubusercontent.com/74184204/162751883-5a9e4e62-8a0f-446a-b10b-19826b540b45.png)

Polish sample:

![image](https://user-images.githubusercontent.com/74184204/162751349-af075f47-c44c-43e0-8cca-044e42e41e9a.png)

## Methods used
* Dynamic reading multiple data files from a various number of directories
* Calculating vectors of letter proportions in each language (ascii characters only)
* Implementing a single-layer neural network of k perceptrons (where k = number of languages) from scratch
* Libraries used for data loading and preprocessing: pandas, numpy
* OOP and clean code
* Unit testing with pytest
