### Začneme len nejakým príkladom

# Import packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
_ = plt.hist(versicolor_petal_length, bins = n_bins)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()

#######################################################

## Swarmplot - je to niečo ako jitter plot 

# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x = "species", y = "petal length (cm)", data = df)

# Label the axes
plt.xlabel("species")
plt.ylabel("petal length")

# Show the plot
plt.show()

#######################################################

## ECDF - vlastná funkcia

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
   
    # POZNAMKA: np.arange(1,10) vráti np.array([1,2,....9])
   
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)                    #zoradenie

    # y-data for the ECDF: y
    y = np.arange(1, len(x) + 1) / n
    
    return x, y

# Compute ECDF for versicolor data: x_vers, y_versx
x_vers, y_vers  = ecdf(versicolor_petal_length)

# Generate plot
plt.plot(x_vers,    y_vers, marker  = ".", linestyle = "none")

# Label the axes
plt.xlabel("Petal length")
plt.ylabel("ECDF")

# Display the plot
plt.show()

## Rozbité po species 

# Compute ECDFs
x_set , y_set  = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
plt.plot(x_set, y_set, marker = ".", linestyle = "none")

plt.plot(x_vers, y_vers, marker = ".", linestyle = "none")

plt.plot(x_virg, y_virg, marker = ".", linestyle = "none")
# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

############################
### PERCENTILY 
############################

# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, q = percentiles)

# Print the result
print(ptiles_vers)

# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle = "none")

# Show the plot
plt.show()

####################################################

## BOXPLOT:  species je stlpec s 3 hodnotami, t.j. sú to TIDY dáta

# Create box plot with Seaborn's default settings
sns.boxplot(x = "species", y = "petal length (cm)", data = df)

plt.xlabel("species")
plt.ylabel("petal length (cm)")
plt.show()

####################################################

# COVARIANCE MATRIX 

# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]

# Print the length/width covariance
print(petal_cov)

########################################
## CORRELATION MATRIX 

def pearson_r(x,y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)

# Print the result
print(r)

####################################################
## BINOMICKE ROZDELENIE 

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n = 100, p = 0.05, size = 10000)

# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels
plt.plot(x,y, marker = ".", linestyle = "none")
plt.xlabel("number of defaults")
plt.ylabel("ecdf")

# Show the plot
plt.show()

####################################################
## POISSON DISTRIBUTION 

### info: je to limitne rozdelenie pre (n) velké a (p) malé 

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size = 10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p

n = [20,100,1000]
p = [0.5, 0.1, 0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n=n[i],p = p[i],size = 10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))


### Príklad na no hitter baseball (iba 251 krat sa to stalo za 115 rokov)

# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size = 10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / len(n_nohitters)

# Print the result
print('Probability of seven or more no-hitters:', p_large)

###############################################################################################

### NORMAL DISTRIBUTION

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, size = 100000)
samples_std3 = np.random.normal(20, 3, size = 100000)
samples_std10 = np.random.normal(20, 10, size = 100000)

# Make histograms
plt.hist(samples_std1, bins = 100, normed = True, histtype = "step")
plt.hist(samples_std3, bins = 100,  normed = True, histtype = "step")
plt.hist(samples_std10, bins = 100,  normed = True, histtype = "step")

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()


## BELMONT STAKES - HORSE COMPETITION 

# Compute mean and standard deviation: mu, sigma
mu, sigma = (np.mean(belmont_no_outliers), np.std(belmont_no_outliers))

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size = 10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x,y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()


#############################################################
#### EXPONENTIAL DISTRIBUTION
############

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)
    
    return t1 + t2
    
# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size = 100000)

# Make the histogram
plt.hist(waiting_times, bins = 100, normed = True, histtype = "step")

# Label axes
plt.xlabel("waiting time")
plt.ylabel("probability")

# Show the plot
plt.show()
s    

