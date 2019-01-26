#############
#### 1. Chapter 
#############


#####################################
## 1. Chapter: K-means clustering  ##
#####################################

kmeans(x, centers = 5, nstart = 20)   # nstart - kolko krat sa ma vykonat 

x <- matrix(rnorm(1000), ncol = 2)
plot(x)

# Create the k-means model: km.out
km.out <- kmeans(x, centers = 3, nstart = 20)

# Inspect the result
summary(km.out)

# Print the cluster membership component of the model
km.out$cluster

# Print the km.out object
print(km.out)


# Scatter plot of x
plot(x, col = km.out$cluster, main = "k-means with 3 clusters", xlab = "",ylab = "")

# Pozor na jednu vec, pokial spustime viac krat kmeans, mozeme dostat rovnaku segmentaciu do clustrov, avsak labelky sa mozu vymenit

# Na vyhodnotenie # clustrov, pouzijeme laktovy graf. x = #clustrov, y = total within sum of squares (from centroids)


# Set up 2 x 3 plotting grid
par(mfrow = c(2, 3))

# Set seed
set.seed(1)

for(i in 1:6) {
  # Run kmeans() on x with three clusters and one start
  km.out <- kmeans(x, centers = 3, nstart = 1)
  
  # Plot clusters
  plot(x, col = km.out$cluster, 
       main = km.out$tot.withinss, 
       xlab = "", ylab = "")
}




# Initialize total within sum of squares error: wss
wss <- 0  

# For 1 to 15 cluster centers
for (i in 1:15) {
  km.out <- kmeans(x, centers = i, nstart = 20)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss
}

# Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

# Set k equal to the number of clusters corresponding to the elbow location
k <- 2 


## Pokemon dataset 
pokemon <- read.csv("https://assets.datacamp.com/production/course_1815/datasets/Pokemon.csv")


# Initialize total within sum of squares error: wss
wss <- 0

# Look over 1 to 15 possible clusters
for (i in 1:15) {
  # Fit the model: km.out
  km.out <- kmeans(pokemon, centers = i, nstart = 20, iter.max = 50)
  # Save the within cluster sum of squares
  wss[i] <- km.out$tot.withinss
}

# Produce a scree plot
plot(1:15, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

# Select number of clusters
k <- 2

# Build model with k clusters: km.out
km.out <- kmeans(pokemon, centers = 2, nstart = 20, iter.max = 50)

# View the resulting model
km.out

# Plot of Defense vs. Speed by cluster membership
plot(pokemon[, c("Defense", "Speed")],
     col = km.out$cluster,
     main = paste("k-means clustering of Pokemon with", k, "clusters"),
     xlab = "Defense", ylab = "Speed")


#############################################
#### 2. Chapter - Hierarchial clustering  ###
#############################################

x <- data.frame(a = c(1:2,10,0), b = c(2,5,11,2))
dist(x) # funkcia ktora vrati vsetky euklidovske vzdialenosti


# Create hierarchical clustering model: hclust.out
hclust.out <- hclust(d = dist(x))

# Inspect the result
summary(hclust.out)



# Cut by height
cutree(hclust.out, h = 7) # odreze pre vysku = 7 (euklidovska)

# Cut by number of clusters
cutree(hclust.out, k = 3) # odreze pre pocet clustrov 


#### Meranie vzdialenosti medzi clustermi

# 1) Complete method - defaul v R. Pozre sa na najvyssiu mozno vzdialenost medzi jedincami v dvoch clustroch
# 2) Single method  - velmi podobne 1.tke avsak neberie sa najdlhsia vzdialenost, ale najkratsia medzi clustermi
# 3) Average method - pouziva sa priemerna vzdialenost 
# 4! Centroid - najde centoridy clustrov a porovnava vzidalenost medzi -- zle spravanie. Niekedy centroidy clustrov maju mensiu vzdialenost ako samotne inner-clustrove vzdialenosti

# Cluster using complete linkage: hclust.complete
hclust.complete <- hclust(dist(x), method = "complete")

# Cluster using average linkage: hclust.average
hclust.average <- hclust(dist(x), method = "average")

# Cluster using single linkage: hclust.single
hclust.single <- hclust(dist(x), method = "single")

# Plot dendrogram of hclust.complete
plot(hclust.complete, main = "Complete")

# Plot dendrogram of hclust.average
plot(hclust.average, main = "Average")

# Plot dendrogram of hclust.single
plot(hclust.single, main = "Single")


# Balanced trees su tvorene complete a average linkage-ami 
"Right! Whether you want balanced or unbalanced trees for your hierarchical clustering model depends on the context 
of the problem you're trying to solve. Balanced trees are essential if you want an even number of observations 
assigned to each cluster. On the other hand, if you want to detect outliers, for example, an unbalanced tree is 
more desirable because pruning an unbalanced tree can result in most observations assigned to one cluster and only
a few observations assigned to other clusters."

## Je potrebna normalizacia dat pri pouzivani unsupervised algov.

# View column means
colMeans(pokemon)

# View column standard deviations
apply(pokemon, 2, sd)

# Scale the data
pokemon.scaled <- scale(pokemon)

# Create hierarchical clustering model: hclust.pokemon
hclust.pokemon <- hclust(dist(pokemon.scaled), method = "complete")

#####
## Segmentacia pozorovani podla hierarchickeho clusteringu
#####

# Apply cutree() to hclust.pokemon: cut.pokemon
cut.pokemon <- cutree(hclust.pokemon, k = 3)  
# zobere rozdelenie do 3 clustrov a olabeluje data
# Je mozne zadat ah parameter h miesto k, co podla mna rozdeluje na zaklade vzdialenosti

# Compare methods
table(cut.pokemon, km.pokemon$cluster)

    1   2   3
1 342 242 204
2   1   1   9
3   0   0   1

"Looking at the table, it looks like the hierarchical clustering model assigns most of the observations 
to cluster 1, while the k-means algorithm distributes the observations relatively evenly among all clusters.
It's important to note that there's no consensus on which method produces better clusters. The job of the 
analyst in unsupervised clustering is to observe the cluster assignments and make a judgment call as to which 
method provides more insights into the data. Excellent job!"


########################################################
#### 3. Chapter - Dimensionality reduction with PCA  ###
########################################################

# Jednoduchy priklad bol 2D kde sme mali y = x + noise
# Jedna sa o 2D priklad, kde s PCA vieme dostat "1D". 1.PCA je regresna priamka a PCA scores su kolme vzdialenosti na tuto priamku

# Perform scaled PCA: pr.out
pr.out <- prcomp(pokemon, scale = TRUE)

# Inspect model output
summary(pr.out)

# Atribut pr.out je aj rotation: the directions of the principal component vectors in terms of the original 

#####
## Vizualizacia pomocou bibplot(pr.out)
## Biplot nanesie data na x = PC1 a y = PC2 suradnice. Premenne ktore maju velmi podobny smer v tejto vizualizacii su korelovane

# Variability of each principal component: pr.var
pr.var <- pr.out$sdev^2

# Variance explained by each principal component: pve
pve <- pr.var / sum(pr.var)


# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")


#######
## Poukazanie na dolezitost scalingu

# Mean of each variable
colMeans(pokemon)

# Standard deviation of each variable
apply(pokemon, 2, sd)

# PCA model with scaling: pr.with.scaling
pr.with.scaling <- prcomp(pokemon, center = TRUE, scale = TRUE)

# PCA model without scaling: pr.without.scaling
pr.without.scaling <- prcomp(pokemon, center = TRUE, scale = FALSE)


# Create biplots of both for comparison
biplot(pr.with.scaling)
biplot(pr.without.scaling)


####################################################
#### 4. Chapter - ALL TOGETHER WITH CASE STUDY   ###
####################################################

url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_1903/datasets/WisconsinCancer.csv"

# Download the data: wisc.df
wisc.df <- read.csv(url)

# Convert the features of the data: wisc.data
wisc.data <- as.matrix(wisc.df[,3:32])

# Set the row names of wisc.data
row.names(wisc.data) <- wisc.df$id

# Create diagnosis vector
diagnosis <- as.numeric(wisc.df$diagnosis == "M" )

# Check column means and standard deviations
colMeans(wisc.data)
apply(wisc.data, 2, sd)

# Execute PCA, scaling if appropriate: wisc.pr
wisc.pr <- prcomp(wisc.data, scale = TRUE, center = TRUE)

# Look at summary of results
summary(wisc.pr)


# Create a biplot of wisc.pr
biplot(wisc.pr)

# Scatter plot observations by components 1 and 2
plot(wisc.pr$x[, c(1, 2)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC2")

# Repeat for components 1 and 3
plot(wisc.pr$x[, c(1, 3)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC3")

# Do additional data exploration of your choosing below (optional)
"Excellent work! Because principal component 2 explains more variance in the original data 
than principal component 3, you can see that the first plot has a cleaner cut separating the two subgroups."

# Set up 1 x 2 plotting grid
par(mfrow = c(1, 2))

# Calculate variability of each component
pr.var <- wisc.pr$sdev^2

# Variance explained by each principal component: pve
pve <- pr.var / sum(pr.var)

# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")







# Scale the wisc.data data: data.scaled
data.scaled <- scale(wisc.data)

# Calculate the (Euclidean) distances: data.dist
data.dist <- dist(data.scaled)

# Create a hierarchical clustering model: wisc.hclust
wisc.hclust <- hclust(data.dist, method = "complete")


# Cut tree so that it has 4 clusters: wisc.hclust.clusters -- pri vzdialenosti 20 to bolo tak (bolo vidno z plotu)
wisc.hclust.clusters <- cutree(wisc.hclust, h = 20)

# Compare cluster membership to actual diagnoses
table(wisc.hclust.clusters, diagnosis)




# Create a hierarchical clustering model: wisc.pr.hclust
wisc.pr.hclust <- hclust(dist(wisc.pr$x[, 1:7]), method = "complete")

# Cut model into 4 clusters: wisc.pr.hclust.clusters
wisc.pr.hclust.clusters <- cutree(wisc.pr.hclust, k = 4)

# Compare to actual diagnoses
table(wisc.pr.hclust.clusters, diagnosis)

# Compare to k-means and hierarchical
table(wisc.km$cluster, diagnosis)
table(wisc.hclust.clusters, diagnosis)
