# Identification of genome domain types based on genomic functional and structural features

This pipeline annotates genomic regions in a domain-scale based on both functional and structural features. Structural features are embedded in
high-throughput chromosome conformation capture (Hi-C) data that captures 3D interactions between genomic regions, and is expressed as a 
genome-wide 2D matrix. This problem requires a framework that jointly clusters the node attributes and neighborhood information given a graph. We first 
embed the graph information into structural features, and then jointly cluster functional and structural features together. We compare our pipeline
with two other aggregative approaches: graph-based regularization (GBR) and Hidden Markov Random Field (HMRF). 

# Data processing

* 1D genomic signals 
* Hi-C data
