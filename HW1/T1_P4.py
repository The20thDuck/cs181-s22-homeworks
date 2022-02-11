#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
# basis function is a transformation applied to the data
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
        
    x = xx.reshape(-1, 1)
    if part == "a":
        unbiased = np.power(np.arange(1, 6), x)

    if part == "b":
        unbiased = np.exp(-(x - np.arange(1960, 2015, 5))**2 /25)

    if part == "c":
        unbiased = np.cos(x / np.arange(1, 6))

    if part == "d":
        unbiased = np.cos(x / np.arange(1, 26))
    return np.concatenate((np.ones(x.shape), unbiased), axis = 1)


# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w


# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

# TODO: plot and report sum of squared error for each basis


for part in 'abcd':
    # grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
    X = make_basis(years, part=part)
    w = find_weights(X, Y)
    
    grid_X = make_basis(grid_years, part=part)
    grid_Yhat  = np.dot(grid_X, w)

    print(f"Sum of squares, {part}:", sum((Y - np.dot(X, w))**2))

    # Plot the data and the regression line.
    plt.plot(years, republican_counts, 'o')
    plt.plot(grid_years, grid_Yhat, '-', label=f'Type: {part}')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.legend()
plt.savefig("basis_regression.png")
plt.show()

for part in 'acd':
    l_1985 = years < 1985
    is_years=False
    X = make_basis(sunspot_counts[l_1985], part=part, is_years=is_years)
    w = find_weights(X, republican_counts[l_1985])
    grid_sunspots = np.linspace(np.min(sunspot_counts), np.max(sunspot_counts), 200)

    grid_X = make_basis(grid_sunspots, part=part, is_years=is_years)
    grid_Yhat  = np.dot(grid_X, w)


    print(f"Sum of squares, {part} (< 1985): {sum((republican_counts[l_1985] - np.dot(make_basis(sunspot_counts[l_1985], part=part, is_years=is_years), w))**2):e}")
    print(f"Sum of squares, {part}         : {sum((republican_counts - np.dot(make_basis(sunspot_counts, part=part, is_years=is_years), w))**2):e}")

    # Plot the data and the regression line.
    plt.plot(sunspot_counts, republican_counts, 'o')
    plt.plot(grid_sunspots, grid_Yhat, '-', label=f'Type: {part}')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans")
    plt.legend()
    plt.savefig(f"basis_regression_sunspot_{part}.png")
    plt.show()