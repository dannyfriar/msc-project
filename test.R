library(data.table)
df <- read.csv("data/domains_clean.csv")

## Get count by SIC code
df <- data.table(subset(df, select=c('vert_code', 'url')))
df <- df[df$vert_code <= 69203, ]
df <- df[df$vert_code >= 69101, ]
df <- df[complete.cases(df), ]
urls <- as.character(df$url)
company_urls <- urls

df1 <- read.csv("rl_toy_implementation/data/first_hop_links.csv", header=FALSE)
df2 <- read.csv("rl_toy_implementation/data/second_hop_links.csv", header=FALSE)
df3 <- read.csv("rl_toy_implementation/data/third_hop_links.csv", header=FALSE)
urls <- c(urls, as.character(df1$V1), as.character(df2$V1), as.character(df3$V1))
rm(df1, df2, df3)

urls[grep("pol", urls)]  # privacy_policy/cookie_policy
urls[grep("batea", urls)] # due to probateaccountants.co.uk having several links from their page (both internal and external)
urls[grep("riva", urls)]  # again from privacy policy
urls[grep("tea", urls)]  # in a lot of cases from "meet the team"
urls[grep("slaw", urls)]  # from e.g. clarkslaw, menzieslaw



##
reward_urls <- read.csv("rl_toy_implementation/results/reward_pages.csv")
reward_urls <- as.character(reward_urls$rewards_pages)
unique(sub("/.*$","", reward_urls))



##
vert_counts <- unique(df[, count_companies:=.N, by=vert_code])
df <- subset(df, select=c('company_name', 'url'))
df <- data.table(df)[order(url)]


