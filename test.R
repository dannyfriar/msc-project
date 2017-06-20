library(data.table)
df <- read.csv("data/domains_clean.csv")

## Get count by SIC code
df <- data.table(subset(df, select=c('vert_code')))
df <- df[df$vert_code <= 69203, ]
df <- df[df$vert_code >= 69101, ]
df <- df[complete.cases(df), ]
vert_counts <- unique(df[, count_companies:=.N, by=vert_code])


##
df <- df[complete.cases(df), ]
df <- subset(df, select=c('company_name', 'url'))
df <- data.table(df)[order(url)]