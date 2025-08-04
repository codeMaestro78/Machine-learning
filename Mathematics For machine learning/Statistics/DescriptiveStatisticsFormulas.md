Descriptive Statistics
Overview
This document provides a reference for key descriptive statistics concepts and their formulas, including types of data, measures of central tendency, measures of dispersion, skewness, and kurtosis. The formulas are presented in LaTeX using dollar signs for proper rendering in Markdown viewers that support LaTeX (e.g., GitHub, Visual Studio Code with Markdown+Math extension, or Jupyter Notebook).
Types of Data

Qualitative Data: Non-numerical data that describes categories or qualities (e.g., colors, names, or labels).
Quantitative Data: Numerical data representing quantities or amounts, divided into:
Discrete: Countable values (e.g., number of students).
Continuous: Measurable values within a range (e.g., height, weight).



Measures of Central Tendency
Mean (Arithmetic Average)
The sum of all data points divided by the number of data points.Formula:$$ \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n} $$Where:  

$ \bar{x} $: Sample mean  
$ x_i $: Individual data point  
$ n $: Number of data points

Median
The middle value when data points are arranged in ascending order.  

For odd $ n $: Median = Middle value  
For even $ n $: Median = $ \frac{\text{Sum of two middle values}}{2} $

Mode
The value that appears most frequently in the dataset. A dataset may have:  

One mode (unimodal)  
Multiple modes (bimodal, multimodal)  
No mode (if no value repeats)

Measures of Dispersion
Range
The difference between the maximum and minimum values.Formula:$$ \text{Range} = \max(x_i) - \min(x_i) $$
Variance
The average of the squared differences from the mean.  

Population Variance:Formula:$$ \sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N} $$  
Sample Variance:Formula:$$ s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1} $$Where:  
$ \sigma^2 $: Population variance  
$ s^2 $: Sample variance  
$ \mu $: Population mean  
$ \bar{x} $: Sample mean  
$ N $: Population size  
$ n $: Sample size

Standard Deviation
The square root of the variance, measuring the spread of data points.  

Population Standard Deviation:Formula:$$ \sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}} $$  
Sample Standard Deviation:Formula:$$ s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}} $$

Interquartile Range (IQR)
The range of the middle 50% of the data, calculated as the difference between the third quartile (Q3) and the first quartile (Q1).Formula:$$ \text{IQR} = Q3 - Q1 $$Where:  

$ Q1 $: 25th percentile (first quartile)  
$ Q3 $: 75th percentile (third quartile)

Skewness
Measures the asymmetry of the data distribution.Formula:$$ \text{Skewness} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^3 / n}{s^3} $$Where:  

$ s $: Sample standard deviation  
Positive skew: Tail on the right (higher values)  
Negative skew: Tail on the left (lower values)  
Zero skew: Symmetric distribution

Kurtosis
Measures the "tailedness" or peakedness of the data distribution.Formula:$$ \text{Kurtosis} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^4 / n}{s^4} - 3 $$Where:  

$ s $: Sample standard deviation  
Subtracting 3 normalizes the value (excess kurtosis) relative to a normal distribution  
Positive kurtosis: Heavy tails, sharp peak (leptokurtic)  
Negative kurtosis: Light tails, flat peak (platykurtic)  
Zero kurtosis: Normal distribution (mesokurtic)

Usage

Use this document as a reference for statistical analysis in research, data science, or academic studies.  
The formulas are formatted in LaTeX with dollar signs for rendering in compatible Markdown viewers (e.g., GitHub, Visual Studio Code with Markdown+Math extension, or Jupyter Notebook).  
These can be implemented in programming languages like Python (using libraries like NumPy or SciPy), R, or statistical software.

Notes

Distinguish between population and sample formulas (e.g., variance and standard deviation) based on your dataset.  
Skewness and kurtosis calculations assume sample data; adjust denominators for population data if needed.  
For practical implementation, statistical libraries in Python (e.g., NumPy, SciPy) or R can compute these metrics efficiently.  
If formulas do not render correctly, ensure your Markdown viewer supports LaTeX (e.g., GitHub, VS Code with a math extension). Alternatively, request a plain text or PDF version.
