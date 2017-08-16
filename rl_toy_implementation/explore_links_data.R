library(data.table)

links_df <- data.table(read.csv('~/links_dataframe.csv'))


first_hop_links <- links_df[type=='first-hop-link']
first_hop_links_uk <- first_hop_links[grepl('.uk', url)]
