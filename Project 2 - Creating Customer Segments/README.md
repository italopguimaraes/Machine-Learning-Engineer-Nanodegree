# Machine Learning Engineer Nanodegree
# Unsupervised Learning
## Project: Creating Customer Segments

# Machine Learning Engineer Nanodegree
# Unsupervised Learning
## Project: Creating Customer Segments

## Project Overview
In this project, unsupervised learning techniques will be applied to product spend data collected by customers from a wholesale distributor in Lisbon, Portugal, to identify customer segments hidden in the data. First, the data are explored by selecting a small subset to be sampled and determining if any product categories are highly correlated with each other. Subsequently, the data were preprocessed, scaling each product category and then identifying (and removing) the unwanted discrepancy values. With good and clean customer expense data, PCA transformations were applied to data and clustering algorithms were implemented to segment customer transformed data. Finally, we compare the segmentation found with additional labeling and consider ways in which this information could assist the wholesale distributor with future service changes.

## Project Highlights
This project is designed to provide the student with hands-on experience with unsupervised learning and work on developing conclusions for a prospect in a real-world dataset. Many companies currently collect large amounts of data about customers and customers and have a strong desire to understand the meaningful relationships that are hidden in their customer base. Being equipped with this information can help the company design future products and services that best meet the demands or needs of its customers.

Things the student will learn completing this project:

- How to apply preprocessing techniques, such as resource sizing and exception detection.
- How to interpret data points that have been scaled, transformed or reduced from PCA.
- How to analyze the dimensions of the PCA and build a new resource space.
- How to optimally group a dataset to find hidden patterns in a dataset.
- How to evaluate the information provided by the cluster data and use it in a meaningful way.

## Description
A wholesale distributor recently tested a change in their method of delivery for some customers, from a five-day morning delivery service to a cheaper overnight delivery service three days a week. The initial tests did not reveal significant unsatisfactory results, so they implemented the cheaper option for all customers. Almost immediately, the distributor began to receive complaints about the change of delivery service and customers were canceling deliveries, losing more money to the distributor than what was being saved. The goal of the project is to help the wholesale distributor find out what types of customers they have to help them make better and more informed business decisions in the future. The task is to use unsupervised learning techniques to see if there are similarities between customers and how to better target customers in different categories.

This project is part of Udacity's machine learning nanodegree program, If you are interested in seeing the project proposal, look in the `projects/customer_segments` directory in this [repository](https://github.com/udacity/br-machine-learning.git).

For more details see [project description](project_description.md)

### Install Software Dependencies

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Code

Template code is provided in the `customer_segments.ipynb` notebook file. You will also be required to use the included `visuals.py` Python file and the `customers.csv` dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. Note that the code included in `visuals.py` is meant to be used out-of-the-box and not intended for students to manipulate. If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

### Run

In a terminal or command window, navigate to the top-level project directory `Project 2 - Creating Customer Segments/` (that contains this README) and run one of the following commands:

```bash
ipython notebook customer_segments.ipynb
```  
or
```bash
jupyter notebook customer_segments.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

## Data

The customer segments data is included as a selection of 440 data points collected on data found from clients of a wholesale distributor in Lisbon, Portugal. More information can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).

Note (m.u.) is shorthand for *monetary units*.

**Features**
1) `Fresh`: annual spending (m.u.) on fresh products (Continuous); 
2) `Milk`: annual spending (m.u.) on milk products (Continuous); 
3) `Grocery`: annual spending (m.u.) on grocery products (Continuous); 
4) `Frozen`: annual spending (m.u.) on frozen products (Continuous);
5) `Detergents_Paper`: annual spending (m.u.) on detergents and paper products (Continuous);
6) `Delicatessen`: annual spending (m.u.) on and delicatessen products (Continuous); 
7) `Channel`: {Hotel/Restaurant/Cafe - 1, Retail - 2} (Nominal)
8) `Region`: {Lisbon - 1, Oporto - 2, or Other - 3} (Nominal) 

## Evaluation

The project was evaluated according to the following [rubric](https://review.udacity.com/#!/rubrics/454/view)

## license
 
For more information see:

[license](LICENSE.txt)
