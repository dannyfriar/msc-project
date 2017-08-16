library(zoo)
library(grid)
library(scales)
library(xtable)
library(ggplot2)
library(gridExtra)
library(data.table)

setwd("~/Desktop/CSML/project/msc-project/uk_web_application")

##--------------------- Find unique fraction of URLs that the crawler got
## UK reward URLs (451)
dt_reward <- data.table(read.csv("../rl_toy_implementation/new_data/company_urls.csv"))
dt_reward$domain <- sapply(dt_reward$url, function(x) sub("www.","", x))
dt_reward$domain <- sapply(dt_reward$domain, function(x) sub("http://","", x))
dt_reward$domain <- sapply(dt_reward$domain, function(x) sub("https://","", x))
dt_reward <- dt_reward[grepl(".uk", url)]
setkey(dt_reward, "domain")

## Held out UK reward URLs (1422)
df_reward_ho <- data.table(read.csv("../data/domains_clean.csv"))
df_reward_ho <- subset(df_reward_ho, select=c('vert_code', 'url'))
df_reward_ho <- df_reward_ho[vert_code <= 69203 & vert_code >= 69101]
df_reward_ho <- df_reward_ho[complete.cases(df_reward)]
dt_reward_ho <- data.table(df_reward_ho)
dt_reward_ho$domain <- sapply(dt_reward_ho$url, function(x) sub("www.","", x))
dt_reward_ho$domain <- sapply(dt_reward_ho$domain, function(x) sub("http://","", x))
dt_reward_ho$domain <- sapply(dt_reward_ho$domain, function(x) sub("https://","", x))
dt_reward_ho <- dt_reward_ho[grepl('.uk', url)]
setkey(dt_reward_ho, "domain")
dt_reward_ho <- merge(dt_reward_ho, dt_reward, all.x=TRUE)  # left join the data tables
dt_reward_ho <- dt_reward_ho[is.na(url.y)]
dt_reward_ho <- subset(dt_reward_ho, select=c(domain, url.x))
setnames(dt_reward_ho, 'url.x', 'url')

## File paths
path_ending <- "all_urls.csv"
feature_weights_path <- "feature_coefficients.csv"
save_results_ending <- "dqn_vs_random.png"
save_coefs_ending <- "coef_weights.png"

results_datatable <- function(folder, filename) {
  df <- data.table(read.csv(paste0(folder, filename), header=FALSE))
  if (grepl("random", folder))
    names(df) <- c('url', 'reward')
  else if (grepl("classifier", folder) | grepl("async", folder))
    names(df) <- c('url', 'reward', 'is_terminal')
  else
    names(df) <- c('url', 'reward', 'is_terminal', 'loss')
  df$step <- 1:nrow(df)
  df$domain <- sapply(df$url, function(x) sub("/.*$","", x))
  df$domain <- sapply(df$domain, function(x) sub("www.","", x))
  df$domain <- sapply(df$domain, function(x) sub("http://","", x))
  df$domain <- sapply(df$domain, function(x) sub("https://","", x))
  df[grepl("}", df$domain)]$domain <- ""
  
  df <- merge(df, dt_reward, by="domain", all.x=TRUE)
  df[is.na(vert_code)]$reward <- 0
  df <- subset(df, select=-c(url.y, vert_code))
  setkey(df, step)
  
  unique_rewards <- subset(df[reward==1], select=c(step, domain))
  unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
  unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)
  
  unique_rewards$domain <- NULL
  setkey(unique_rewards, step)
  df <- merge(df, unique_rewards, all.x=TRUE)
  df[1]$unique_reward_count <- ifelse(is.na(df[1]$unique_reward_count), 0, 1)
  df$unique_reward_count <- na.locf(df$unique_reward_count)
  df$cum_reward <- cumsum(df$reward)
  return(df)
}

r_all_urls <- results_datatable("results/random_crawler_results/", path_ending)
r_all_urls$type <- "Random"
r_all_urls$is_terminal <- NA; r_all_urls$loss <- NA
linear_all_urls <- results_datatable("results/linear_dqn_results/", path_ending)
linear_all_urls$type <- "Q-learning"
l_buff_all_urls <- results_datatable("results/linear_buffer_results/", path_ending)
l_buff_all_urls$type <- "Q-learning+buffer"
text_linear_urls <- results_datatable("results/text_linear_results/", path_ending)
text_linear_urls$type <- "Q-learning+text"
l_penalty_urls <- results_datatable("results/linear_penalty_results/", path_ending)
l_penalty_urls$type <- "Q-learning+penalty"
embedding <- results_datatable("results/embedding_results/", path_ending)
embedding$type <- "DQN+embedding"
classifier_urls <- results_datatable("results/classifier_results/", "all_urls_crawler.csv")
classifier_urls$type <- "Classifier"
classifier_urls$loss <- NA
async <- results_datatable("results/async_results/", "all_urls_revisit.csv")
async$type <- "Async"
async$loss <- NA

## Combine and plot unique rewards
g_uniq <- ggplot(data=results_df, aes(x=step, y=unique_reward_count, color=type))
g_uniq <- g_uniq + geom_line(size=0.9) + labs(x='Pages Crawled', y='Unique Rewards', color="")
g_uniq <- g_uniq + theme(legend.position="top")
g_uniq <- g_uniq + scale_x_continuous(labels=comma) + scale_y_continuous()
g_uniq <- g_uniq + guides(color=guide_legend(nrow=2,byrow=TRUE))
g_uniq


# ##------------------- Multiple runs performance plots
# multiple_runs_datatable <- function(folder, filename) {
#   df <- data.table(read.csv(paste0(folder, filename), header=FALSE))
#   if (grepl("random", folder))
#     names(df) <- c('url', 'reward', 'run')
#   else if (grepl("classifier", folder) | grepl("async", folder))
#     names(df) <- c('url', 'reward', 'is_terminal', 'run')
#   else
#     names(df) <- c('url', 'reward', 'is_terminal', 'loss', 'run')
#   
#   df$step <- c(1:200000, 1:200000, 1:200000, 1:200000, 1:200000)
#   df$domain <- sapply(df$url, function(x) sub("/.*$","", x))
#   df$domain <- sapply(df$domain, function(x) sub("www.","", x))
#   df$domain <- sapply(df$domain, function(x) sub("http://","", x))
#   df$domain <- sapply(df$domain, function(x) sub("https://","", x))
#   df[grepl("}", df$domain)]$domain <- ""  
#   
#   df <- merge(df, dt_reward, by="domain", all.x=TRUE)
#   df[is.na(vert_code)]$reward <- 0
#   df <- subset(df, select=-c(url.y, vert_code))
#   setkeyv(df, c('step', 'run'))
#   
#   unique_rewards <- subset(df[reward==1], select=c(step, domain, run))
#   unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=c(domain, run)))]
#   unique_rewards$simple_count <- 1
#   unique_rewards$unique_reward_count <- with(unique_rewards, ave(simple_count, simple_count, run, FUN=seq_along))
#   unique_rewards$domain <- NULL
#   unique_rewards$simple_count <- NULL
#   
#   setkeyv(unique_rewards, c('step', 'run'))
#   df <- merge(df, unique_rewards, all.x=TRUE)
#   df[step==1]$unique_reward_count <- ifelse(is.na(df[step==1]$unique_reward_count), 0, 1)
#   df <- df[, unique_r_count := na.locf(unique_reward_count), by=run]
#   
#   plot_df <- subset(df, select=c(step, run, unique_r_count))
#   plot_df <- plot_df[, mean_reward := mean(unique_r_count), by=step]
#   plot_df <- plot_df[, min_reward := min(unique_r_count), by=step]
#   plot_df <- plot_df[, max_reward := max(unique_r_count), by=step]
#   plot_df <- plot_df[, stdev := sd(unique_r_count), by=step]
#   
#   print(paste0(folder, filename))
#   print(mean(plot_df[step == max(plot_df$step)]$unique_r_count))
#   print(sd(plot_df[step == max(plot_df$step)]$unique_r_count))
#   
#   plot_df <- unique(subset(plot_df, select=c(step, mean_reward, min_reward, max_reward, stdev)))
#   return(plot_df)
# }
# 
# random_urls <- multiple_runs_datatable("results/random_crawler_results/", path_ending)
# random_urls$type <- "Random"
# linear_urls <- multiple_runs_datatable("results/linear_dqn_results/", path_ending)
# linear_urls$type <- "Q-learning"
# embed_urls <- multiple_runs_datatable("results/embedding_results/", path_ending)
# embed_urls$type <- "DQN+embedding"
# class_urls <- multiple_runs_datatable("results/classifier_results/", "all_urls_crawler.csv")
# class_urls$type <- "Classifier"
# 
# plot_df <- rbind(class_urls, async_plot, embed_urls, random_urls)
# g <- ggplot(data=plot_df, aes(x=step, y=mean_reward, color=type)) + geom_line(size=0.9)
# # g <- g + geom_ribbon(aes(ymin=mean_reward+stdev, ymax=mean_reward-stdev, fill=type), alpha=0.35, color=NA)
# # g <- g + geom_errorbar(aes(ymax=mean_reward+stdev, ymin=mean_reward-stdev))
# g <- g + labs(x='Pages Crawled', y='Unique Rewards Found', color='')
# g <- g + theme(legend.position="top") + guides(color=guide_legend(nrow=2), fill=FALSE)
# g + scale_x_continuous(label=comma)





##-------------------- Distribution of number of links in the web graph
uk_web_links <- data.table(read.csv("../uk_web_application/building_data/test_results/uk_links.csv"))
# rl_web_links <- rl_web_links[num_links<=500]
g_uk_links <- ggplot(data=uk_web_links, aes(x=num_links)) + geom_histogram(color='firebrick', fill='lightblue')
g_uk_links <- g_uk_links + labs(x='Number of Outgoing Links', y='Count')
g_uk_links
summary(uk_web_links$num_links)


##-------------------- Results on full UK web
uk_random <- results_datatable("../uk_web_application/results/random_results/", "all_urls.csv")
uk_random$type <- "Random"
uk_random$is_terminal <- NA; uk_random$loss <- NA

# ##Combine and plot rewards
# uk_results_df <- rbind(async, classifier_urls, embedding, r_all_urls)

## Combine and plot unique rewards
g_uk <- ggplot(data=uk_random, aes(x=step, y=unique_reward_count, color=type))
g_uk <- g_uk + geom_line(size=0.9) + labs(x='Pages Crawled', y='Unique Rewards', color="")
g_uk <- g_uk + theme(legend.position="top")
g_uk <- g_uk + scale_x_continuous(labels=comma) + scale_y_continuous()
g_uk <- g_uk + guides(color=guide_legend(nrow=2,byrow=TRUE))
g_uk


