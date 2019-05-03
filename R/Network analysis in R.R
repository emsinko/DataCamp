library(tidyverse)


# Network Data: Adjacent matrix
# Network Data: Edgelist : 2 stlpce (od - do), vrcholy medzi ktorymi je spojenie 

# Load igraph
library(igraph)



friends <- read.csv("https://assets.datacamp.com/production/course_4474/datasets/friends.csv")

# Inspect the first few rows of the dataframe 'friends'
head(friends)

# Convert friends dataframe to a matrix
friends.mat <- as.matrix(friends)

# Convert friends matrix to an igraph object
g <- graph.edgelist(friends.mat, directed = FALSE)


# Make a very basic plot of the network
plot(g)

# Subset vertices and edges
V(g)  # print vertices 
E(g)  # print edges

# Count number of edges
gsize(g)
ecount(g) # ekvivalent


 # Count number of vertices
gorder(g)
vcount(g)  # ekvivalent


####
## Atributes ----
####

g  # print: attr: name (v/c)  --> name as a "Vertices" attribute
V(g)$name

genders <- c("M", "F", "F", "M", "M", "M", "F", "M", "M", "F", "M", "F", "M", "F", "M", "M")
ages  <- c(18, 19, 21, 20, 22, 18, 23, 21, 22, 20, 20, 22, 21, 18, 19, 20)


# Create new vertex attribute called 'gender'
g <- set_vertex_attr(g, "gender", value = genders)  # ekvivalent (ak sa nemylim) V(g)$gender <- genders

# Create new vertex attribute called 'age'
g <- set_vertex_attr(g, "age", value = ages) # ekvivalent (ak sa nemylim) V(g)$age <- ages


# View all vertex attributes in a list
vertex_attr(g)

# View attributes of first five vertices in a dataframe
V(g)[[1:5]] 
#V(g)[1:5]  # ine zobrazenie 


##
# Edge attributes and subsetting
##

hours <- c(1, 2, 2, 1, 2, 5, 5, 1, 1, 3, 2, 1, 1, 5, 1, 2, 4, 1, 3, 1, 1, 1, 4, 1, 3, 3, 4)

# Create new edge attribute called 'hours'
g <- set_edge_attr(g, "hours", value = hours)


# View edge attributes of graph object
edge_attr(g)

# Find all edges that include "Britt"
E(g)[[inc('Britt')]]  

# Find all pairs that spend 4 or more hours together per week
E(g)[[hours >= 4]]  # vrati zoznam


##
# Visualizing attributes
##

friends1_edges <- read.csv("https://assets.datacamp.com/production/course_4474/datasets/friends1_edges.csv")
friends1_nodes <- read.csv("https://assets.datacamp.com/production/course_4474/datasets/friends1_nodes.csv")

friends1_edges
friends1_nodes

# Create an igraph object with attributes directly from dataframes
g1 <- graph_from_data_frame(d = friends1_edges, vertices = friends1_nodes, directed = FALSE)
# param: d = A data frame containing a symbolic edge list in the first two columns. Additional columns are considered as edge attributes.

# Subset edges greater than or equal to 5 hours
E(g1)[[hours >= 5]]  

# Set vertex color by gender
V(g1)$color <- ifelse(V(g1)$gender == "F", "orange", "dodgerblue")

# Plot the graph
plot(g1, vertex.label.color = "black")


##
# Styling vertices
# params:  size, labels, colors(fill), shape, edgest(thickness / type)

##
# Choosing layouts
# implemented rules: minimize egde crossing , do not allow vertices to overlap,  make edges length as uniform as possible, symmetry of network

library(igraph)

# Plot the graph object g1 in a circle layout
plot(g1, vertex.label.color = "black", layout = layout_in_circle(g1))

# Plot the graph object g1 in a Fruchterman-Reingold layout 
plot(g1, vertex.label.color = "black", layout = layout_with_fr(g1))

# Plot the graph object g1 in a Tree layout 
m <- layout_as_tree(g1)
plot(g1, vertex.label.color = "black", layout = m)

# Plot the graph object g1 using igraph's chosen layout 

m1 <- layout_nicely(g1)  # igraph skusi vybrat sam 
plot(g1, vertex.label.color = "black", layout = m1)


##
# Visualizing edges ---
##

# Create a vector of weights based on the number of hours each pair spend together
w1 <- E(g1)$hours

# Plot the network varying edges by weights
m1 <- layout_nicely(g1)
plot(g1, 
     vertex.label.color = "black", 
     edge.color = 'black',
     edge.width = w1,
     layout = m1)


# Create a new igraph object by deleting edges that are less than 2 hours long 
g2 <- delete_edges(g1, E(g1)[hours < 2])


# Plot the new graph 
w2 <- E(g2)$hours
m2 <- layout_nicely(g2)

plot(g2, 
     vertex.label.color = "black", 
     edge.color = 'black',
     edge.width = w2,
     layout = m2)


##
# Quiz: How many edges does Jasmine have in the network g1?
## 

E(g1)[[inc("Jasmine")]]


##
# Directed igraph objects ---
## 

measles <- read.csv("https://assets.datacamp.com/production/course_4474/datasets/measles.csv")

# Get the graph object
g <- graph_from_data_frame(measles, directed = TRUE)

# is the graph directed?
is.directed(g)

# Is the graph weighted?
is.weighted(g)

# Where does each edge originate from?
table(head_of(g, E(g)))

##
#Identifying edges for each vertex ---
##

# Make a basic plot
plot(g, 
     vertex.label.color = "black", 
     edge.color = 'gray77',
     vertex.size = 0,
     edge.arrow.size = 0.1,
     layout = layout_nicely(g))

# Is there an edge going from vertex 184 to vertex 178?
g['184', '178']

# Is there an edge going from vertex 178 to vertex 184?
g['178', '184']

# Show all edges going to or from vertex 184
incident(g, '184', mode = c("all"))

# Show all edges going out from vertex 184
incident(g, '184', mode = c("out")) # moze byt este all/in/out

