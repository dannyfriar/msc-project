library(data.table)
library(ggplot2)
library(scales)

# Read in data
companies.df <- data.table(read.csv('../data/domains_clean.csv'))
companies.df$X <- NULL

# Count by vert code
vert_count <- subset(companies.df, select=c(vert_code, vert_desc))
vert_count <- unique(vert_count[, count:=.N, by=list(vert_code)])
vert_count <- vert_count[order(-count)]
