library(data.table)
setwd("/home/daniel/msc-project/rl_toy_implementation/building_data")

#--------------- Read data
# All company pages
links_df <- data.table(read.csv("../data/links_dataframe.csv"))
company_urls <- subset(links_df[hops==0], select=c(url))
company_urls$type <- "company-url"

# First hop links
first_hop <- data.table(read.csv("../new_data/first_hop_links.csv"))
first_hop$type <- "first-hop-link"

# Actual reward pages
arp <- data.table(read.csv("../new_data/actual_reward_pages.csv"))
arp$type <- "outgoing-first-hop"

# First hop outgoing
out_first_hop <- data.table(read.csv("../new_data/first_hop_outgoing_uk_links.csv"))
out_first_hop$type <- "outgoing-first-hop"

# Second hop links
second_hop <- data.table(read.csv("../new_data/second_hop_links.csv"))
second_hop$type <- "second-hop-link"

# Second hop outgoing links
out_second_hop <- data.table(read.csv("../new_data/second_hop_outgoing_uk_links.csv"))
out_second_hop$type <- "outgoing-second-hop"

#--------------- Combine data
all_urls <- rbind(company_urls, first_hop, arp, out_first_hop, second_hop, out_second_hop)
all_urls$domain <- sub("/.*$","", all_urls$url)
write.csv(all_urls, "../new_data/links_dataframe.csv", row.names=FALSE)





