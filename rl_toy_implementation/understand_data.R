##------- Understanding data

## Company URLs
df <- read.csv("../data/domains_clean.csv")
df <- data.table(subset(df, select=c('vert_code', 'url')))
df <- df[df$vert_code <= 69203, ]
df <- df[df$vert_code >= 69101, ]
df <- df[complete.cases(df), ]
df_reward <- df

company_urls <- as.character(df$url)
company_urls <- gsub("http://", "", company_urls)
company_urls <- gsub("https://", "", company_urls)
length(company_urls)
grep("/", company_urls) ## check they're all domains

## First, second and third hop links
first_hop_df <- read.csv("data/first_hop_links.csv", header=FALSE)
first_hop_links <- setdiff(first_hop_df$V1, company_urls)
length(first_hop_links)

second_hop_df <- read.csv("data/second_hop_links.csv", header=FALSE)
second_hop_links <- setdiff(second_hop_df$V1, c(company_urls, first_hop_links))
length(second_hop_links)

third_hop_df <- read.csv("data/third_hop_links.csv", header=FALSE)
third_hop_links <- setdiff(third_hop_df$V1, c(company_urls, first_hop_links, second_hop_links))
length(third_hop_links)

all_links <- c(first_hop_links, second_hop_links, third_hop_links, company_urls)

first_hop_links <- sub("/.*$","", first_hop_links)
second_hop_links <- sub("/.*$","", second_hop_links)
third_hop_links <- sub("/.*$","", third_hop_links)

all_hop_links <- c(first_hop_links, second_hop_links, third_hop_links)
all_hop_links <- unique(all_hop_links)

## Intersection between these sets
length(intersect(first_hop_links, company_urls))
length(intersect(second_hop_links, company_urls))
length(intersect(third_hop_links, company_urls))
length(intersect(all_hop_links, company_urls))

## Save this domains data
save_links_df <- data.frame(url=all_hop_links)
write.csv(save_links_df, "data/domains_list.csv", row.names=FALSE)

## Internet domains i.e. .com etc
domains <- sub('.*\\.', '', all_hop_links)
domains_df <- data.table(data.frame(domains=domains))
domains_df <- unique(domains_df[, count:=.N, by=domains])
domains_df <- domains_df[order(-count)]
domains_df <- domains_df[1:50]
domains_df$ending <- paste0(".", as.character(domains_df$domains))
domains_df <- subset(domains_df, select=c(ending))
write.csv(domains_df, "data/domains_endings.csv", row.names=FALSE)

## Links dataframe
links_df <- data.frame(url=all_links)
links_df$hops <- c(rep(1, length(first_hop_links)), rep(2, length(second_hop_links)), rep(3, length(third_hop_links)), rep(0, length(company_urls)))
links_df$domain <- sub("/.*$","", links_df$url)
write.csv(links_df, "data/links_dataframe.csv", row.names=FALSE)


links_df <- links_df[links_df$hops==0, ]
links_df$hops <- NULL
links_df$domain <- NULL
write.csv(links_df, "~/links_data.csv", row.names=FALSE)

