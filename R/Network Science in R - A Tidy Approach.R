##############################################
### Network Science in R - A Tidy Approach ###
##############################################

# Link na lektorovu univerzitny kurz
#http://users.dimi.uniud.it/~massimo.franceschet/ns/syllabus/syllabus.html

library(tidyverse)


# Read the nodes file into the variable nodes
nodes <- read_csv("nodes.csv")
nodes <- read_csv("https://assets.datacamp.com/production/repositories/2048/datasets/92011bee6be869ca8dfc5a6c44a2febf4b6b73b3/nodes.csv")

# Read the ties file into the variable ties
ties <- read_csv("ties.csv")
ties <- read_csv("https://assets.datacamp.com/production/repositories/2048/datasets/afe8ca0e602ec92d73881f1bdf7a02466612c295/ties.csv")

# Print nodes
nodes  # strukture ID a name. Cislo  vrchola a jeho meno

# Print ties
ties  # ties je v strukture: from, to , weight a hovori nam od koho ku komu je spojenie. (ID)


library(igraph)

# Make the network from the data frame ties and print it
g <- graph_from_data_frame(ties, directed = FALSE, vertices = nodes)
g

# Explore the set of nodes
V(g)  # v ako vertices

# Print the number of nodes
vcount(g)

# Explore the set of ties
E(g)  # E ako egde

# Print the number of ties
ecount(g)



# Give the name "Madrid network" to the network 
g$name <- "Madrid network"
g$name

# Add node attribute id and print the node `id` attribute
V(g)$id <- 1:vcount(g)
V(g)$id

# Print the tie `weight` attribute
E(g)$weight

# Print the network and spot the attributes
g

#=============================
#--- Visualizing networks ----
#=============================


# Geometries for nodes have names starting geom_node_. 
# For example, geom_node_point() draws each node as a point. geom_node_text() draws a text label on each node.

# Geometries for ties have names starting geom_edge_. 
# For example, geom_edge_link() draws edges as a straight line between nodes.


library(ggraph)  # This extends ggplot2 with new geometries to visualize the nodes and ties of a network.


# Visualize the network with the Kamada-Kawai layout 
ggraph(g,  layout = "with_kk") + 
  # Add an edge link geometry mapping transparency to weight 
  geom_edge_link(aes( alpha = weight)) +   # prida hrany 
  # Add a node point geometry
  geom_node_point()  + # prida bodky s vrcholmi
  geom_node_text(aes(label = id), repel = TRUE) #Set the repel argument to TRUE to prevent the labels from overlapping.


# Visualize the network in a circular layout
ggraph(g, layout = "in_circle") + 
  # Map tie transparency to its weight
  geom_edge_link(aes(alpha = weight)) + 
  geom_node_point()


####
## Visualize node centrality

# Plot with the Kamada-Kawai layout 
ggraph(g, layout = "with_kk") + 
  # Add an edge link geom, mapping alpha to weight
  geom_edge_link(aes(alpha = weight)) + 
  # Add a node point geom, mapping size to degree
  geom_node_point(aes(size = degree))




# Change the layout so points are on a grid
ggraph(g, layout = "on_grid") + 
  geom_edge_link(aes(alpha = weight)) + 
  

#=============================
#--- Centrality measures  ----
#=============================

# Degree = pocet hran vychadzajucich zu vrcholu
# Strength = vo vazenych networkoch sucet vychadzajucich hran (sucet ich vah) z vrcholu

# betweenness(): a measure that quantifies how often a node lies on the shortest path between other nodes.
#  -- ak som spravne pochopil zoberu sa vsetky dvojice vertexov, vypocita sa:  #pocet najkratsich ciest prechadzajucich cez node / # najkratsich vzdialenosti medzi dvoma node-ami. Tieto podiele sa zosumuju napriec vsetkymi dvoj-combo
#  -- https://en.wikipedia.org/wiki/Betweenness_centrality
#  -- v tomto baliku je to vsak len pocet vyskytov v najkratsich cestach
#  -- v pripade 3 statov (3 clustre networkov) maju najvyssi betweennes prave tí komunikátori, ktorí su prepojení medzi štátmi

#closeness(): a measure that quantifies how close a node is to all other nodes in the network in terms of shortest path distance.

### Degree:
nodes_with_centrality <- nodes %>%
  # Add a column containing the degree of each node
  mutate(degree = degree(g)) %>%
  # Arrange rows by descending degree
  arrange(desc(degree))

# See the result
nodes_with_centrality

### Strength
nodes_with_centrality <- nodes %>%
  mutate(
    degree = degree(g),
    # Add a column containing the strength of each node
    strength = strength(g)
  ) %>%
  # Arrange rows by descending strength
  arrange(desc(strength))

# See the result
nodes_with_centrality

####
## TIES BETWEENNESS


# Calculate the reciprocal of the tie weights
dist_weight <- 1 / E(g)$weight

ties_with_betweenness <- ties %>%
  # Add an edge betweenness column weighted by dist_weight
  mutate(betweenness = edge_betweenness(g, weights = dist_weight))

# Review updated ties
ties_with_betweenness

#### 
# Aby sme mali hrany popisane okrem prechodov cez ID-cka aj podla mien
ties_betweenness <- ties %>% 
  # Left join to the nodes matching 'from' to 'id'
  left_join(nodes, by = c("from" = "id")) %>% 
  # Left join to nodes again, now matching 'to' to 'id'
  left_join(nodes, by = c("to" = "id")) %>%
  # Select only relevant variables
  select(from, to, name_from = name.x, name_to = name.y, betweenness) %>%
  # Arrange rows by descending betweenness
  arrange(desc(betweenness))


####
## Visualize node centrality
####

# Plot with the Kamada-Kawai layout 
ggraph(g, layout = "with_kk") + 
  # Add an edge link geom, mapping alpha to weight
  geom_edge_link(aes(alpha = weight)) + 
  # Add a node point geom, mapping size to degree
  geom_node_point(aes(size = degree))


# Update the previous plot, mapping node size to strength
ggraph(g, layout = "with_kk") + 
  geom_edge_link(aes(alpha = weight)) + 
  geom_node_point(aes(size = strength))
  

ggraph(g, layout = "with_kk") + 
  geom_edge_link(aes(alpha = betweenness)) + 
  # Add a node point geom, mapping size to degree
  geom_node_point(aes(size = degree))

###
## Filter and viisualize important ties !! Filtrovanie priamo v aes  
###

# Calculate the median betweenness
median_betweenness = median(E(g)$betweenness)

ggraph(g, layout = "with_kk") + 
  # Filter ties for betweenness greater than the median
  geom_edge_link(aes(alpha = betweenness, filter = betweenness > median_betweenness)) +  # Toto je sranda, nevedel som. Nezaujimave nam uplne vynecha a spriehladni graf
  theme(legend.position="none")

####
#The strength of weak ties
####

# Weak ties - hrany medzi roznymi komunitami (dolezite "MOSTY" !)
# Tie co maju nizky Strength (toto su vsak weak nodes) = vo vazenych networkoch sucet vychadzajucich hran (sucet ich vah) z vrchol
# Ak sa bavime o weak ties, mozeme hovorit o tych co maju malu weight

# Strong ties - presny opak. 

####
#How many weak ties are there?
####
  
tie_counts_by_weight <- ties %>% 
  # Count the number of rows with each weight
  count(weight) %>%   #group_by(weight) + n()
  # Add a column of the percentage of rows with each weight
  mutate(percentage = 100 * n/ nrow(ties)) 

# See the result
tie_counts_by_weight

###

# Make is_weak TRUE whenever the tie is weak
is_weak <- E(g)$weight == 1 

# Check that the number of weak ties is the same as before
sum(is_weak)

ggraph(g, layout = "with_kk") +
  # Add an edge link geom, mapping color to is_weak
  geom_edge_link(aes(color = is_weak))


#Visualize the network with only weak ties using the filter aesthetic set to the is_weak variable.
ggraph(g, layout = "with_kk") + 
  # Map filter to is_weak
  geom_edge_link(aes(filter = is_weak), alpha = 0.5) 

####
## 
####

# Swap the variables from and to -- aby sme mali duplicitne cesty (obojsmerne)

ties_swapped <- ties %>%
  mutate(temp = to, to = from, from = temp) %>% 
  select(-temp)
ties_bound <- bind_rows(ties, ties_swapped)  # spojenie

# Using ties_bound, plot to vs. from, filled by weight
ggplot(ties_bound, aes(x = from, y = to, fill = factor(weight))) +
  # Add a raster geom
  geom_raster() +
  # Label the color scale as "weight"
  labs(fill = "weight")


####
## Adjacent matrix
####

#Two nodes are adjacent when they are directly connected by a tie. 
#An adjacency matrix contains the details about which nodes are adjacent for a whole network.

# Get the weighted adjacency matrix
A <- as_adjacency_matrix(g, attr = "weight", names = FALSE)

# See the results
A  # vrati sparse maticu nxn (n = pocet vrcholov) a v indexe [s,t] sa nachadza vaha medzi vrcholo s a t (ak nie je vazeny network tak 0/1 predstavujuca ci je spojenie)

# Calculate node strengths as row sums of adjacency
rowSums(A)

# Create an logical adjacency matrix
B <- (A > 0)

# Calculate node degrees as row sums of logical adjacency
rowSums(B)

###
#Pearson correlation coefficient
###

# Compute the Pearson correlation matrix of A
S <- cor(A)  # vypocita korelacie takto: zobere stlpce 3,4, spocita medzi nimi pearsona a korelaciu da do [3,4] & [4,3]

# Set the diagonal of S to 0
diag(S) <- 0

# Flatten S to be a vector
flat_S <- as.vector(S)

# Plot a histogram of similarities
hist(flat_S, xlab = "Similarity", main = "Histogram of similarity")


# Using nodes, plot strength vs.degree
ggplot(nodes, aes(x =  degree, y = strength)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

# Calculate the Pearson correlation coefficient  
cor(nodes$degree, nodes$strength)


###
## Transforming the similarity matrix
###

########################################
## Transformacie:

# graph to matrix:
A <- as_adjacency_matrix((g))  # attr = "weight

# matrix to graph:
g <- graph_from_adjacency_matrix(A)

# graph to data frame 
df <- as_data_frame(g, what = "both")

# data frame to graph 
g <- graph_from_data_frame(df$ties, vertices = df$nodes)

# matrix to data frame 
df = as_data_frame(graph_from_adjacency_matrix(A), what ="both")

# data frame to matrix
A <- as_adjacency_matrix(graph_from_data_frame(df$ties, vertices = df$nodes))

########################################

# Similarity matrix -> S  (pearson coeficients)

h <- graph_from_adjacency_matrix(S, mode = "undirected", weighted = TRUE)
sim_df <- igraph::as_data_frame(h, what = "edges")  # bacha, aj dplyr ma as_data_frame

# Convert sim_df to a tibble
sim_tib <- as_tibble(sim_df)

# See the results
sim_tib

####
####
# The similarity data frame sim contains pairs of nodes and their similarities

sim_joined <- sim %>% 
  # Left join to nodes matching "from" to "id"
  left_join(nodes, by = c("from" = "id")) %>% 
  # Left join to nodes matching "to" to "id", setting suffixes
  left_join(nodes, by = c("to" = "id"), suffix = c("_from","_to"))

# See the results
sim_joined

####
####

sim_joined %>%	
  # Filter for degree from & degree to greater than or equal to 10	
  filter(degree_from >= 10, degree_to >= 10) %>%	
  arrange(desc(similarity))


sim_filtered <- sim_joined %>% 
  # Filter on similarity greater than 0.6
  filter(similarity > 0.6) 

# Convert to an undirected graph
filtered_network <- graph_from_data_frame(sim_filtered, directed = FALSE) 

# Plot with Kamada-Kawai layout
ggraph(filtered_network, layout = "with_kk") + 
  # Add an edge link geom, mapping transparency to similarity
  geom_edge_link(aes(alpha = similarity))


####
## CLUSTERING
####

# Compute a distance matrix = 1 - SIMILARITY MATRIX  (ak su rovnake = corelacia = 1, tak distance je 1-1=0)
D <- 1 - S

# Obtain a distance object 
d <- as.dist(D)

# Run average-linkage clustering method and plot the dendrogram 
cc <- hclust(d, method = "average")
plot(cc)

# Find the similarity of the first pair of nodes that have been merged 
S[40,45]

# NOTE ! cc$merge[1, ] shows that the first pair of nodes to be merged during clustering is the pair (40, 45)

# Cut the dendrogram tree into 4 clusters
cls <- cutree(cc, k = 4)

# Add cluster information to nodes
nodes_with_clusters <- nodes %>% mutate(cluster = cls)

# See the result
nodes_with_clusters

### Analyze clusters:

# Who is in cluster 1?
nodes %>%
  # Filter rows for cluster 1
  filter(cluster == 1 ) %>% 
  # Select the name column
  select(name)

# Calculate properties of each cluster
nodes %>%
  # Group by cluster
  group_by(cluster) %>%
  # Calculate summary statistics
  summarise(
    # Number of nodes
    size = n(), 
    # Mean degree
    avg_degree = mean(degree),
    # Mean strength
    avg_strength = mean(strength)
  ) %>% 
  # Arrange rows by decreasing size
  arrange(desc(size))


###
# Visualize the clusters
###

# From previous step
V(g)$cluster <- nodes$cluster

# Update the plot
ggraph(g, layout = "with_kk") + 
  geom_edge_link(aes(alpha = weight), show.legend=FALSE) +  
  geom_node_point(aes(color = factor(cluster))) + 
  labs(color = "cluster") +
  # Facet the nodes by cluster, with a free scale
  facet_nodes(~cluster, scales = "free")


###########
### INTERACTIVE VIZUALIZATION WITH VISNETWORK
###########

library(visNetwork)
# Convert from igraph to visNetwork
data <- toVisNetworkData(g)

# Print the head of the data nodes
head(data$node)

# ... do the same for the edges (ties)
head(data$edges)

# From previous step
data <- toVisNetworkData(g)

# Visualize the network
visNetwork(nodes = data$nodes, edges = data$edges, width = 300, height = 300)

# Add to the plot
visNetwork(nodes = data$nodes, edges = data$edges, width = 300, height = 300) %>%
  # Set the layout to Kamada-Kawai
  visIgraphLayout(layout = "layout_with_kk")

# See a list of possible layouts
ls("package:igraph", pattern = "^layout_.")

# Update the plot
visNetwork(nodes = data$nodes, edges = data$edges, width = 300, height = 300) %>%
  # Change the layout to be in a circle
  visIgraphLayout(layout = "layout_in_circle")

####
## Pridat interactivitu: ak kliknem na nejaky vrchol, ostanu farebne len jeho susedia
# Add to the plot
visNetwork(nodes = data$nodes, edges = data$edges, width = 300, height = 300) %>%
  # Choose an operator
  visIgraphLayout(layout = "layout_with_kk") %>%
  # Change the options to highlight the nearest nodes and ties
  visOptions(highlightNearest = TRUE)

####
## Interaktivyny vyber podla ID (podmla mien)
####

# Update the plot
visNetwork(nodes = data$nodes, edges = data$edges, width = 300, height = 300) %>%
  visIgraphLayout(layout = "layout_with_kk") %>%
  # Change the options to allow selection of nodes by ID
  visOptions(nodesIdSelection = TRUE)


####
## Interaktivyny vyber podla clustrov
####

# Copy cluster node attribute to color node attribute
V(g)$color <- V(g)$cluster

# Convert g to vis network data
data <- toVisNetworkData(g)

# Update the plot
visNetwork(nodes = data$nodes, edges = data$edges, width = 300, height = 300) %>%
  visIgraphLayout(layout = "layout_with_kk") %>%
  # Change options to select by group
  visOptions(selectedBy = "group")