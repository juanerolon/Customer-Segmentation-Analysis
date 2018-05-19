
# Customer Segmentation Analysis via Unsupervised Learning  

### Juan E. Rolon, 2017

<img src="segments_logo.jpg"
     alt="customer_segments"
     style="float: left; margin-right: 10px; width: 250px;" />

## Project Overview

In this project, I analyze a dataset containing data on various customers' annual spending amounts (reported in monetary units) of diverse product categories for internal structure. 

One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.  

The dataset for this project can be found on the UCI Machine Learning Repository.
For the purposes of this project, the features 'Channel' and 'Region' will be excluded in the analysis — with focus instead on the six product categories recorded for customers.  

This project was submitted as part of the requisites required to obtain Machine Learning Engineer Nanodegree from Udacity. 

### Installation

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Code

- The code of this project is provided in the `customer_segments.ipynb` notebook file.   

- You will also be required to use the included `visuals.py` Python file and the `customers.csv` dataset file to complete your work.   

- The code included in `visuals.py` is meant to be used out-of-the-box; feel free to modify it if you consider so. It produces graphs for soem of the visualizations created in the project ipython notebook.

### Run

In a terminal or command window, navigate to the top-level project directory `customer_segments/` (that contains this README) and run one of the following commands:

```bash
ipython notebook customer_segments.ipynb
```  
or
```bash
jupyter notebook customer_segments.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Datasets

The customer segments data is included as a selection of 440 data points collected on data found from clients of a wholesale distributor in Lisbon, Portugal.   

More information can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).

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


```python

```
