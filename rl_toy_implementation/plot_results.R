library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)
library(scales)
library(zoo)

#-------------------- Read random train results data
# random_results <- read.csv("results/random_crawler_results_backlinks.csv")
random_results <- read.csv("results/random_crawler_results/random_crawler_results_revisit.csv")
random_results$type <- "Random Crawler"
# dqn_results <- read.csv("results/dqn_crawler_train_results_backlinks.csv")
dqn_results <- read.csv("results/linear_dqn_results/dqn_crawler_train_results_revisit.csv")
# dqn_results <- read.csv("results/dqn_crawler_test_results.csv")
dqn_results$type <- "Q-Learning Agent"

df <- rbind(random_results, subset(dqn_results, select=-c(nn_loss)))
# df <- rbind(random_results, dqn_results)
df$type <- factor(df$type)

## Reward plot
g_reward <- ggplot(data=df, aes(x=pages_crawled, y=total_reward, color=type)) 
g_reward <- g_reward + geom_line(size=0.9) + labs(x='Pages Crawled', y='Total Reward', color='')
g_reward <- g_reward + theme(legend.position='top')
g_reward <- g_reward + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)

# ## Terminal states plot
# g_terminal <- ggplot(data=df, aes(x=pages_crawled, y=terminal_states, color=type)) 
# g_terminal <- g_terminal + geom_line(size=0.9) + labs(x='Pages Crawled', y='Number Terminal States', color='')
# g_terminal <- g_terminal + theme(legend.position='top')
# g_terminal <- g_terminal + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)

##--------------------- Find unique fraction of URLs that the crawler got
## DQN crawler
dqn_all_urls <- data.table(read.csv("results/linear_dqn_results/all_urls.csv", header=FALSE))
names(dqn_all_urls) <- c('url', 'reward', 'is_terminal')
dqn_all_urls$step <- 1:nrow(dqn_all_urls)
dqn_all_urls$cum_reward <- cumsum(dqn_all_urls$reward)
dqn_all_urls$domain <- sapply(dqn_all_urls$url, function(x) sub("/.*$","", x))
setkey(dqn_all_urls, step)

unique_rewards <- subset(dqn_all_urls[reward==1], select=c(step, domain))
unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)

unique_rewards$domain <- NULL
setkey(unique_rewards, step)
dqn_all_urls <- merge(dqn_all_urls, unique_rewards, all.x=TRUE)
dqn_all_urls[1]$unique_reward_count <- ifelse(is.na(dqn_all_urls[1]$unique_reward_count), 0, 1)
dqn_all_urls$unique_reward_count <- na.locf(dqn_all_urls$unique_reward_count)
dqn_all_urls$is_terminal <- NULL

# Random crawler
r_all_urls <- data.table(read.csv("results/random_crawler_results/random_all_urls.csv", header=FALSE))
names(r_all_urls) <- c('url', 'reward')
r_all_urls$step <- 1:nrow(r_all_urls)
r_all_urls$cum_reward <- cumsum(r_all_urls$reward)
r_all_urls$domain <- sapply(r_all_urls$url, function(x) sub("/.*$","", x))
setkey(r_all_urls, step)

unique_rewards <- subset(r_all_urls[reward==1], select=c(step, domain))
unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)

unique_rewards$domain <- NULL
setkey(unique_rewards, step)
r_all_urls <- merge(r_all_urls, unique_rewards, all.x=TRUE)
r_all_urls[1]$unique_reward_count <- ifelse(is.na(r_all_urls[1]$unique_reward_count), 0, 1)
r_all_urls$unique_reward_count <- na.locf(r_all_urls$unique_reward_count)

# Combine and plot
dqn_all_urls$type <- "Q-learning Agent"
r_all_urls$type <- "Random Crawler"
unique_results_df <- rbind(dqn_all_urls, r_all_urls)
g_uniq <- ggplot(data=unique_results_df, aes(x=step, y=unique_reward_count, color=type))
g_uniq <- g_uniq + geom_line(size=0.9) + labs(x='Pages Crawled', y='Unique Rewards')
g_uniq <- g_uniq + theme(legend.position="top")
g_uniq <- g_uniq + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)



##-------------------------------------- Combine plots
# Share legend function
grid_arrange_shared_legend <- function(..., nrow = 1, ncol = length(list(...)), position = c("bottom", "right")) {
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position = "none"))
  gl <- c(gl, nrow = nrow, ncol = ncol)

  combined <- switch(position,
                     "bottom" = arrangeGrob(do.call(arrangeGrob, gl),
                                            legend,
                                            ncol = 1,
                                            heights = unit.c(unit(1, "npc") - lheight, lheight)),
                     "right" = arrangeGrob(do.call(arrangeGrob, gl),
                                           legend,
                                           ncol = 2,
                                           widths = unit.c(unit(1, "npc") - lwidth, lwidth)))
  grid.newpage()
  grid.arrange(combined)

}

# Combine
g <- grid_arrange_shared_legend(g_reward, g_uniq, nrow = 1, ncol = 2)
g
ggsave(filename="../figures/our_work/dqn_vs_random.png", plot=g, width = 15, height = 10, units = "cm")

##----------------------- Look at feature coefficients
feature_coefs <- data.table(read.csv("results/linear_dqn_results/feature_coefficients.csv"))
feature_coefs$coef_mag <- abs(feature_coefs$coef)
feature_coefs <- feature_coefs[order(-coef_mag)]
feature_coefs$words <- factor(feature_coefs$words, levels=unique(feature_coefs$words))

g_coef <- ggplot(data=feature_coefs[1:25], aes(x=words, y=coef, fill=words)) + geom_bar(stat='identity')
g_coef <- g_coef + theme(legend.position='none', axis.text.x=element_text(angle=45, hjust=1))
g_coef <- g_coef + labs(x="Word", y="Weight")
g_coef


##-------------------- Examine presence of particular words (find URLs that contain these)
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






##-------------------- Read accounts df - debugging
accounts_df <- read.csv('results/account_results.csv')

##-------------------- Read all URLs - debugging
all_urls <- read.csv('results/all_urls.csv', header=FALSE)

