library(zoo)
library(grid)
library(scales)
library(xtable)
library(ggplot2)
library(gridExtra)
library(data.table)

setwd("~/Desktop/CSML/project/msc-project/rl_toy_implementation")
revisit = TRUE

##--------------------- Find unique fraction of URLs that the crawler got
## Load companies data URLs
df_reward <- data.table(read.csv("../data/domains_clean.csv"))
df_reward <- subset(df_reward, select=c('vert_code', 'url'))
df_reward <- df_reward[vert_code <= 69203 & vert_code >= 69101]
df_reward <- df_reward[complete.cases(df_reward)]

if (revisit == TRUE) {
  path_ending <- "all_urls_revisit.csv"
  feature_weights_path <- "feature_coefficients_revisit.csv"
  save_results_ending <- "dqn_vs_random_revisit.png"
  save_coefs_ending <- "coef_weights_revisit.png"
} else {
  path_ending <- "all_urls.csv"
  feature_weights_path <- "feature_coefficients.csv"
  save_results_ending <- "dqn_vs_random.png"
  save_coefs_ending <- "coef_weights.png"
}

## Linear DQN crawler
dqn_all_urls <- data.table(read.csv(paste0("results/linear_dqn_results/", path_ending), header=FALSE))
names(dqn_all_urls) <- c('url', 'reward', 'is_terminal')
dqn_all_urls$step <- 1:nrow(dqn_all_urls)
dqn_all_urls$domain <- sapply(dqn_all_urls$url, function(x) sub("/.*$","", x))
dqn_all_urls$domain <- sapply(dqn_all_urls$domain, function(x) sub("www.","", x))
dqn_all_urls$domain <- sapply(dqn_all_urls$domain, function(x) sub("http://","", x))
dqn_all_urls$domain <- sapply(dqn_all_urls$domain, function(x) sub("https://","", x))
dqn_all_urls[grepl("}", dqn_all_urls$domain)]$domain <- ""
dt_reward <- data.table(df_reward)
dt_reward$domain <- sapply(dt_reward$url, function(x) sub("www.","", x))
dt_reward$domain <- sapply(dt_reward$domain, function(x) sub("http://","", x))
dt_reward$domain <- sapply(dt_reward$domain, function(x) sub("https://","", x))
dqn_all_urls <- merge(dqn_all_urls, dt_reward, by="domain", all.x=TRUE)
dqn_all_urls[is.na(vert_code)]$reward <- 0
dqn_all_urls <- subset(dqn_all_urls, select=-c(url.y, vert_code))
setkey(dqn_all_urls, step)

unique_rewards <- subset(dqn_all_urls[reward==1], select=c(step, domain))
unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)

unique_rewards$domain <- NULL
setkey(unique_rewards, step)
dqn_all_urls <- merge(dqn_all_urls, unique_rewards, all.x=TRUE)
dqn_all_urls[1]$unique_reward_count <- ifelse(is.na(dqn_all_urls[1]$unique_reward_count), 0, 1)
dqn_all_urls$unique_reward_count <- na.locf(dqn_all_urls$unique_reward_count)
dqn_all_urls$cum_reward <- cumsum(dqn_all_urls$reward)
dqn_all_urls$type <- "Q-learning Agent"

## Random crawler
r_all_urls <- data.table(read.csv(paste0("results/random_crawler_results/", path_ending), header=FALSE))
names(r_all_urls) <- c('url', 'reward')
r_all_urls$step <- 1:nrow(r_all_urls)
r_all_urls$domain <- sapply(r_all_urls$url, function(x) sub("/.*$","", x))
r_all_urls$domain <- sapply(r_all_urls$url, function(x) sub("/.*$","", x))
r_all_urls$domain <- sapply(r_all_urls$domain, function(x) sub("www.","", x))
r_all_urls$domain <- sapply(r_all_urls$domain, function(x) sub("http://","", x))
r_all_urls$domain <- sapply(r_all_urls$domain, function(x) sub("https://","", x))
r_all_urls[grepl("}", r_all_urls$domain)]$domain <- ""
r_all_urls <- merge(r_all_urls, dt_reward, by="domain", all.x=TRUE)
r_all_urls[is.na(vert_code)]$reward <- 0
r_all_urls <- subset(r_all_urls, select=-c(url.y, vert_code))
setkey(r_all_urls, step)

unique_rewards <- subset(r_all_urls[reward==1], select=c(step, domain))
unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)

unique_rewards$domain <- NULL
setkey(unique_rewards, step)
r_all_urls <- merge(r_all_urls, unique_rewards, all.x=TRUE)
r_all_urls[1]$unique_reward_count <- ifelse(is.na(r_all_urls[1]$unique_reward_count), 0, 1)
r_all_urls$unique_reward_count <- na.locf(r_all_urls$unique_reward_count)
r_all_urls$cum_reward <- cumsum(r_all_urls$reward)
r_all_urls$is_terminal <- NA
r_all_urls$type <- "Random Crawler"

## Deep DQN crawler (with target net + buffer)
deep_all_urls <- data.table(read.csv(paste0("results/buffer_dqn_results/", path_ending), header=FALSE))
names(deep_all_urls) <- c('url', 'reward', 'is_terminal')
deep_all_urls$step <- 1:nrow(deep_all_urls)
deep_all_urls$domain <- sapply(deep_all_urls$url, function(x) sub("/.*$","", x))
deep_all_urls$domain <- sapply(deep_all_urls$domain, function(x) sub("www.","", x))
deep_all_urls$domain <- sapply(deep_all_urls$domain, function(x) sub("http://","", x))
deep_all_urls$domain <- sapply(deep_all_urls$domain, function(x) sub("https://","", x))
deep_all_urls[grepl("}", deep_all_urls$domain)]$domain <- ""
deep_all_urls <- merge(deep_all_urls, dt_reward, by="domain", all.x=TRUE)
deep_all_urls[is.na(vert_code)]$reward <- 0
deep_all_urls <- subset(deep_all_urls, select=-c(url.y, vert_code))
setkey(deep_all_urls, step)

unique_rewards <- subset(deep_all_urls[reward==1], select=c(step, domain))
unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)

unique_rewards$domain <- NULL
setkey(unique_rewards, step)
deep_all_urls <- merge(deep_all_urls, unique_rewards, all.x=TRUE)
deep_all_urls[1]$unique_reward_count <- ifelse(is.na(deep_all_urls[1]$unique_reward_count), 0, 1)
deep_all_urls$unique_reward_count <- na.locf(deep_all_urls$unique_reward_count)
deep_all_urls$cum_reward <- cumsum(deep_all_urls$reward)
deep_all_urls$type <- "DQN Agent"

## Q-learning + buffer
l_buff_all_urls <- data.table(read.csv(paste0("results/linear_buffer_results/", path_ending), header=FALSE))
names(l_buff_all_urls) <- c('url', 'reward', 'is_terminal')
l_buff_all_urls$step <- 1:nrow(l_buff_all_urls)
l_buff_all_urls$domain <- sapply(l_buff_all_urls$url, function(x) sub("/.*$","", x))
l_buff_all_urls$domain <- sapply(l_buff_all_urls$domain, function(x) sub("www.","", x))
l_buff_all_urls$domain <- sapply(l_buff_all_urls$domain, function(x) sub("http://","", x))
l_buff_all_urls$domain <- sapply(l_buff_all_urls$domain, function(x) sub("https://","", x))
l_buff_all_urls[grepl("}", l_buff_all_urls$domain)]$domain <- ""
l_buff_all_urls <- merge(l_buff_all_urls, dt_reward, by="domain", all.x=TRUE)
l_buff_all_urls[is.na(vert_code)]$reward <- 0
l_buff_all_urls <- subset(l_buff_all_urls, select=-c(url.y, vert_code))
setkey(l_buff_all_urls, step)

unique_rewards <- subset(l_buff_all_urls[reward==1], select=c(step, domain))
unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=domain))]
unique_rewards$unique_reward_count <- 1:nrow(unique_rewards)

unique_rewards$domain <- NULL
setkey(unique_rewards, step)
l_buff_all_urls <- merge(l_buff_all_urls, unique_rewards, all.x=TRUE)
l_buff_all_urls[1]$unique_reward_count <- ifelse(is.na(l_buff_all_urls[1]$unique_reward_count), 0, 1)
l_buff_all_urls$unique_reward_count <- na.locf(l_buff_all_urls$unique_reward_count)
l_buff_all_urls$cum_reward <- cumsum(l_buff_all_urls$reward)
l_buff_all_urls$type <- "Q-learning+Buffer"

## Combine and plot rewards
results_df <- rbind(l_buff_all_urls, dqn_all_urls, r_all_urls)
# results_df <- rbind(dqn_all_urls, r_all_urls)
# results_df <- rbind(deep_all_urls, r_all_urls)
g_reward <- ggplot(data=results_df, aes(x=step, y=cum_reward, color=type))
g_reward <- g_reward + geom_line(size=0.9) + labs(x='Pages Crawled', y='Total Reward', color="")
g_reward <- g_reward + theme(legend.position='top')
g_reward <- g_reward + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)

## Combine and plot unique rewards
g_uniq <- ggplot(data=results_df, aes(x=step, y=unique_reward_count, color=type))
g_uniq <- g_uniq + geom_line(size=0.9) + labs(x='Pages Crawled', y='Unique Rewards', color="")
g_uniq <- g_uniq + theme(legend.position="top")
g_uniq <- g_uniq + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)

##-------------------------------------- Combine plots function
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
if (revisit == TRUE) {
  g
  ggsave(filename=paste0("../figures/our_work/", save_results_ending), plot=g, width=15, height=10, units="cm")
} else {
  g_uniq
  ggsave(filename=paste0("../figures/our_work/", save_results_ending), plot=g_uniq, width=15, height=10, units="cm")
}

##----------------------- Look at feature coefficients
feature_coefs <- data.table(read.csv(paste0("results/linear_buffer_results/", feature_weights_path)))
feature_coefs$coef_mag <- abs(feature_coefs$coef)
feature_coefs <- feature_coefs[order(-coef_mag)]
feature_coefs$words <- factor(feature_coefs$words, levels=unique(feature_coefs$words))

g_coef <- ggplot(data=feature_coefs[1:20], aes(x=words, y=coef, fill=words)) + geom_bar(stat='identity')
g_coef <- g_coef + theme(legend.position='none', axis.text.x=element_text(angle=45, hjust=1))
g_coef <- g_coef + labs(x="", y="Weight")
g_coef
ggsave(filename=paste0("../figures/our_work/", save_coefs_ending), plot=g_coef, width=15, height=10, units="cm")

# Look at some words that may be expected
expected_words <- c("account", "tax", "law", "legal")
regex_string <- paste(expected_words, collapse="|")
chosen_features <- feature_coefs[grepl(regex_string, feature_coefs$words)]
chosen_features <- subset(chosen_features[!grepl("[[:digit:]]", chosen_features$words)], select=-c(coef_mag))
setcolorder(chosen_features, c('words', 'coef'))
colnames(chosen_features) <- c("Word", "Coefficient")
View(chosen_features)
print(xtable(chosen_features, caption="Weights of Common Words", label="common_word_table"), include.rownames=FALSE)

##----------------------- Slope
results_df <- rbind(dqn_all_urls, r_all_urls)
# filtered_results_df <- results_df[step < 5000]
filtered_results_df <- results_df[step %% 750 == 0 | step == 1]
filtered_results_df$change_reward <- diff(c(0, filtered_results_df$cum_reward))
filtered_results_df[change_reward<=0]$change_reward <- 0
# filtered_results_df <- filtered_results_df[2:nrow(filtered_results_df)]
g_reward_slope <- ggplot(data=filtered_results_df, aes(x=step, y=change_reward, color=type)) + geom_line()
# g_reward_slope <- g_reward_slope + geom_smooth(alpha=0.2, linetype=2)
g_reward_slope <- g_reward_slope + labs(x="Pages Crawled", y="Average Reward per Timestep")
g_reward_slope = g_reward_slope + theme(legend.position="bottom", legend.title=element_text(""))
# g_reward_slope <- g_reward_slope + facet_grid(type~.)
g_reward_slope

##---------------------- Testing predicted values vs actual values
# predicted_values <- data.table(read.csv("results/linear_dqn_results/test_value_revisit.csv"))
predicted_values <- data.table(read.csv("results/buffer_dqn_results/test_value_revisit.csv"))
actual_values <- data.table(read.csv("results/linear_dqn_results/actual_value_revisit.csv"))
setkey(predicted_values, url); setkey(actual_values, url)
test_values <- merge(predicted_values, actual_values)
test_values$domain <- sapply(test_values$url, function(x) sub("www.","", x))
test_values$domain <- sapply(test_values$domain, function(x) sub("http://","", x))
test_values$domain <- sapply(test_values$domain, function(x) sub("https://","", x))
test_values <- merge(test_values, dt_reward, by="domain", all.x=TRUE)
test_values <- test_values[!(is.na(vert_code) & true_value==1)]
test_values <- subset(test_values, select=-c(vert_code, url.y, domain))
setnames(test_values, 'url.x', 'url')
test_values <- test_values[true_value > 0][order(-true_value)]
test_values <- test_values[value >= 0 & value <=1]
test_values$true_value <- factor(test_values$true_value, levels=unique(test_values$true_value),
                                 labels=c("1", "gamma", "gamma^2", "gamma^3"))
test_values <- test_values[, median_value:=median(value), by=true_value]
g_value <- ggplot(data=test_values, aes(x=value, fill=true_value)) + geom_density(alpha=0.6)
g_value <- g_value + geom_vline(aes(xintercept=median_value, group=true_value), linetype=2)
g_value <- g_value + labs(x="Predicted Value Function", y="", fill="True Value Function")
g_value <- g_value + theme(legend.position="bottom") + guides(fill=guide_legend(nrow=2))
g_value <- g_value + facet_grid(true_value~., scales = "free_y") + guides(fill=FALSE, color=FALSE)
g_value



