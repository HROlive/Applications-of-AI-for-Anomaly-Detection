![Course](images/banner.jpg)

## Table of Contents
1. [Description](#description)
2. [Information](#information)
3. [File descriptions](#files)
4. [Certificate](#certificate)

<a name="descripton"></a>
## Description

Whether your organization needs to monitor cybersecurity threats, fraudulent financial transactions, product defects, or equipment health, artificial intelligence can help catch data abnormalities before they impact your business. AI models can be trained and deployed to automatically analyze datasets, define “normal behavior,” and identify breaches in patterns quickly and effectively. These models can then be used to predict future anomalies. With massive amounts of data available across industries and subtle distinctions between normal and abnormal patterns, it’s critical that organizations use AI to quickly detect anomalies that pose a threat.

In this workshop, we learned how to identify anomalies and failures in time-series data, estimate the remaining useful life of the corresponding parts, and map anomalies to failure conditions. More specifically, how to prepare time-series data for AI model training, develop an XGBoost ensemble tree model, build a deep learning model using a long short-term memory (LSTM) network, and create an autoencoder that detects anomalies for predictive maintenance. At the end of the workshop, we are able to use AI to estimate the condition of equipment and predict when maintenance should be performed.

<a name="information"></a>
## Information

The overall goals of this course were the following:
> - Prepare data and build, train, and evaluate models using XGBoost, autoencoders, and GANs;
> - Detect anomalies in datasets with both labeled and unlabeled data;
> - Classify anomalies into multiple categories regardless of whether the original data was labeled.

More detailed information and links for the course can be found on the [course website](https://www.nvidia.com/en-us/training/instructor-led-workshops/anomaly-detection/).

<a name="files"></a>
## File descriptions

The description of the files in this repository can be found bellow:
- 1 - Anomaly Detection in Network Data Using GPU-Accelerated XGBoost:
  - [Lab1-XGBoost-For-Timeseries](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab1-XGBoost-For-Timeseries.ipynb) - Notebook;
  - [Lab1-Presentation_202208](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab1-Presentation_202208.pptx) - Slides;
<br></br>
  > Learn how to detect anomalies using supervised learning:
    > - Prepare data for GPU acceleration using the provided dataset.
    > - Train a binary and multi-class classifier using the popular machine learning algorithm XGBoost.
    > - Assess and improve your model’s performance before deployment.
______________
- 2 - Anomaly Detection in Network Data Using GPU-Accelerated Autoencoder:
  - [Lab2-LSTM-For-Timeseries](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab2-LSTM-For-Timeseries.ipynb) - Notebook;
  - [Lab2-Presentation_202208](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab2-Presentation_202208.pptx) - Slides;
<br></br>
  > Learn how to detect anomalies using modern unsupervised learning:
    > - Build and train a deep learning-based autoencoder to work with unlabeled data.
    > - Apply techniques to separate anomalies into multiple classes.
    > - Explore other applications of GPU-accelerated autoencoders.
______________
- 3 - Anomaly Detection in Network Data Using GANs:
  - [Lab3-AE-For-Anomaly-Detection](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab3-AE-For-Anomaly-Detection.ipynb) - Notebook;
  - [Lab3-Presentation_202208](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab3-Presentation_202208.pptx) - Slides;
  - [assignment](https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/assignment.py) - Assessement;
<br></br>
  > Learn how to detect anomalies using GANs:
    > - Train an unsupervised learning model to create new data.
    > - Use that new data to turn the problem into a supervised learning problem.
    > - Compare the performance of this new approach to more established approaches.

<a name="certificate"></a>
## Certificate

The certificate for the workshop can be found bellow:

["Applications of AI for Anomaly Detection" - NVIDIA Deep Learning Institute]() (Issued On: March 2023)