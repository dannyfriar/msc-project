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

dt_reward <- data.table(df_reward)
dt_reward$domain <- sapply(dt_reward$url, function(x) sub("www.","", x))
dt_reward$domain <- sapply(dt_reward$domain, function(x) sub("http://","", x))
dt_reward$domain <- sapply(dt_reward$domain, function(x) sub("https://","", x))

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

results_datatable <- function(folder, filename) {
  df <- data.table(read.csv(paste0(folder, filename), header=FALSE))
  if (grepl("random", folder))
    names(df) <- c('url', 'reward')
  else if (grepl("classifier", folder) | grepl("async", folder))
    names(df) <- c('url', 'reward', 'is_terminal')
  else
    names(df) <- c('url', 'reward', 'is_terminal')
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

# Check domains
domain_rewards <- subset(linear_all_urls, select=c(domain, reward))[, total_reward:=sum(reward), by=domain]
domain_rewards <- unique(domain_rewards[total_reward>0][order(-total_reward)])

## Combine and plot rewards
results_df <- rbind(async, classifier_urls, embedding, r_all_urls)
g_reward <- ggplot(data=results_df, aes(x=step, y=cum_reward, color=type))
g_reward <- g_reward + geom_line(size=0.9) + labs(x='Pages Crawled', y='Total Reward', color="")
g_reward <- g_reward + theme(legend.position='top')
g_reward <- g_reward + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)
g_reward

## Combine and plot unique rewards
g_uniq <- ggplot(data=results_df, aes(x=step, y=unique_reward_count, color=type))
g_uniq <- g_uniq + geom_line(size=0.9) + labs(x='Pages Crawled', y='Unique Rewards', color="")
g_uniq <- g_uniq + theme(legend.position="top")
g_uniq <- g_uniq + scale_x_continuous(labels=comma) + scale_y_continuous()
g_uniq <- g_uniq + guides(color=guide_legend(nrow=2,byrow=TRUE))
g_uniq


##------------------- Multiple runs performance plots
multiple_runs_datatable <- function(folder, filename) {
  df <- data.table(read.csv(paste0(folder, filename), header=FALSE))
  if (grepl("random", folder))
    names(df) <- c('url', 'reward', 'run')
  else if (grepl("classifier", folder) | grepl("async", folder))
    names(df) <- c('url', 'reward', 'is_terminal', 'run')
  else
    names(df) <- c('url', 'reward', 'is_terminal', 'loss', 'run')
  
  df$step <- c(1:200000, 1:200000, 1:200000, 1:200000, 1:200000)
  df$domain <- sapply(df$url, function(x) sub("/.*$","", x))
  df$domain <- sapply(df$domain, function(x) sub("www.","", x))
  df$domain <- sapply(df$domain, function(x) sub("http://","", x))
  df$domain <- sapply(df$domain, function(x) sub("https://","", x))
  df[grepl("}", df$domain)]$domain <- ""  
  
  df <- merge(df, dt_reward, by="domain", all.x=TRUE)
  df[is.na(vert_code)]$reward <- 0
  df <- subset(df, select=-c(url.y, vert_code))
  setkeyv(df, c('step', 'run'))
  
  unique_rewards <- subset(df[reward==1], select=c(step, domain, run))
  unique_rewards <- unique_rewards[!duplicated(subset(unique_rewards, select=c(domain, run)))]
  unique_rewards$simple_count <- 1
  unique_rewards$unique_reward_count <- with(unique_rewards, ave(simple_count, simple_count, run, FUN=seq_along))
  unique_rewards$domain <- NULL
  unique_rewards$simple_count <- NULL
  
  setkeyv(unique_rewards, c('step', 'run'))
  df <- merge(df, unique_rewards, all.x=TRUE)
  df[step==1]$unique_reward_count <- ifelse(is.na(df[step==1]$unique_reward_count), 0, 1)
  df <- df[, unique_r_count := na.locf(unique_reward_count), by=run]
  
  plot_df <- subset(df, select=c(step, run, unique_r_count))
  plot_df <- plot_df[, mean_reward := mean(unique_r_count), by=step]
  plot_df <- plot_df[, min_reward := min(unique_r_count), by=step]
  plot_df <- plot_df[, max_reward := max(unique_r_count), by=step]
  plot_df <- plot_df[, stdev := sd(unique_r_count), by=step]
  
  print(paste0(folder, filename))
  print(mean(plot_df[step == max(plot_df$step)]$unique_r_count))
  print(sd(plot_df[step == max(plot_df$step)]$unique_r_count))
  
  plot_df <- unique(subset(plot_df, select=c(step, mean_reward, min_reward, max_reward, stdev)))
  return(plot_df)
}

random_urls <- multiple_runs_datatable("results/random_crawler_results/", path_ending)
random_urls$type <- "Random"
linear_urls <- multiple_runs_datatable("results/linear_dqn_results/", path_ending)
linear_urls$type <- "Linear Q-learning"
embed_urls <- multiple_runs_datatable("results/embedding_results/", path_ending)
embed_urls$type <- "DQN+embedding"
class_urls <- multiple_runs_datatable("results/classifier_results/", "all_urls_crawler.csv")
class_urls$type <- "Classifier"
async_urls <- multiple_runs_datatable("results/classifier_results/", "all_urls_crawler.csv")
async_urls$type <- "Async"

plot_df <- rbind(class_urls, async_urls, embed_urls, linear_urls, random_urls)
# write.csv(plot_df, "~/plot_df.csv")

g <- ggplot(data=plot_df, aes(x=step, y=mean_reward, color=type)) + geom_line(size=0.9)
g <- g + geom_ribbon(aes(ymin=min_reward, ymax=max_reward, fill=type), alpha=0.35, color=NA)
# g <- g + geom_errorbar(aes(ymax=mean_reward+stdev, ymin=mean_reward-stdev))
g <- g + labs(x='Pages Crawled', y='Unique Rewards Found', color='')
g <- g + theme(legend.position="top") + guides(color=guide_legend(nrow=2), fill=FALSE)
g <- g + scale_x_continuous(label=comma)
# g
ggsave('~/plot.png', plot=g)



##----------------------- Look at feature coefficients
feature_coefs <- data.table(read.csv(paste0("results/linear_dqn_results/", feature_weights_path)))
# feature_coefs <- data.table(read.csv(paste0("results/text_linear_results/", feature_weights_path)))
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
# chosen_features$type <- NULL
setcolorder(chosen_features, c('words', 'coef'))
colnames(chosen_features) <- c("Word", "Coefficient")
View(chosen_features)
print(xtable(chosen_features, caption="Weights of Common Words", label="common_word_table"), include.rownames=FALSE)

##----------------------- Slope
results_df <- rbind(l_buff_all_urls, r_all_urls)
# filtered_results_df <- results_df[step < 5000]
filtered_results_df <- results_df[step %% 2000 == 0 | step == 1]
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
# predicted_values <- data.table(read.csv("results/linear_dqn_results/test_value_revisit.csv", stringsAsFactors=FALSE))
# predicted_values <- data.table(read.csv("results/linear_dqn_results/visited_value.csv", stringsAsFactors=FALSE))
# predicted_values <- data.table(read.csv("results/linear_buffer_results/visited_value.csv", stringsAsFactors=FALSE))
# predicted_values <- data.table(read.csv("results/embedding_results/predicted_value.csv", stringsAsFactors=FALSE))
predicted_values <- data.table(read.csv("results/async_results/predicted_value.csv", stringsAsFactors=FALSE))
# predicted_values <- data.table(read.csv("results/deep_dqn_results/test_value_revisit.csv", stringsAsFactors=FALSE))
# predicted_values <- data.table(read.csv("results/linear_buffer_results/test_value_revisit.csv", stringsAsFactors=FALSE))
# predicted_values <- data.table(read.csv("results/embedding_results/test_value_revisit.csv", stringsAsFactors=FALSE))
# actual_values <- data.table(read.csv("results/actual_value_revisit.csv", stringsAsFactors=FALSE))
# actual_values <- data.table(read.csv("results/embedding_results/actual_value_visited.csv", stringsAsFactors=FALSE))
# actual_values <- data.table(read.csv("results/actual_value_visited.csv", stringsAsFactors=FALSE))
actual_values <- data.table(read.csv("results/async_results/actual_value_visited.csv", stringsAsFactors=FALSE))
setkey(predicted_values, url); setkey(actual_values, url)
test_values <- merge(predicted_values, unique(actual_values))
test_values$domain <- sapply(test_values$url, function(x) sub("www.","", x))
test_values$domain <- sapply(test_values$domain, function(x) sub("http://","", x))
test_values$domain <- sapply(test_values$domain, function(x) sub("https://","", x))
test_values <- merge(test_values, dt_reward, by="domain", all.x=TRUE)
test_values <- test_values[!(is.na(vert_code) & true_value==1)]
test_values <- subset(test_values, select=-c(vert_code, url.y, domain))
setnames(test_values, 'url.x', 'url')
test_values <- test_values[true_value >= 0][order(-true_value)]
test_values <- test_values[, median_value:=median(value), by=true_value]
test_values <- test_values[value >= 0 & value <= 1]
test_values[true_value==0.9]$true_value <- 0.75
test_values[true_value==0.81]$true_value <- 0.75^2
test_values$true_numeric_value <- test_values$true_value
test_values$true_numeric_value <- as.numeric(test_values$true_numeric_value)
test_values$true_value <- factor(test_values$true_value, levels=unique(test_values$true_value),
                                 labels=c("1", "gamma", "gamma^2", "0"))
g_value <- ggplot(data=test_values, aes(x=value, fill=true_value)) + geom_histogram(alpha=0.8, color='grey3')
g_value <- g_value + geom_vline(aes(xintercept=true_numeric_value, group=true_value), linetype=2, size=0.75)
g_value <- g_value + labs(x="Predicted Value Function", y="", fill="True Value Function")
g_value <- g_value + theme(legend.position="bottom") + guides(fill=guide_legend(nrow=2))
g_value <- g_value + facet_grid(true_value~., scales = "free_y") + guides(fill=FALSE, color=FALSE)
g_value


##---------------------- URL value classifier
class_vals <- data.table(read.csv("results/classifier_results/test_value.csv"))
class_vals <- class_vals[order(-value)]
class_vals$numeric_value <- as.numeric(class_vals$value)
class_vals$value <- factor(class_vals$value, levels=unique(class_vals$value))
class_vals <- class_vals[, median_value:=median(predicted_value), by=value]
g_class <- ggplot(data=class_vals, aes(x=predicted_value, fill=value)) + geom_histogram(alpha=0.8, color='grey3')
g_class <- g_class + geom_vline(aes(xintercept=numeric_value, group=value), linetype=2, size=0.75)
g_class <- g_class + labs(x="Predicted Value Function", y="", fill="True Value Function")
g_class <- g_class + theme(legend.position="bottom") + guides(fill=guide_legend(nrow=2))
g_class <- g_class + facet_grid(value~., scales = "free_y") + guides(fill=FALSE, color=FALSE)
g_class + scale_x_continuous(limits=c(0, 1))



##---------------------- BOW value classifier
# Loss plot
bow <- data.table(read.csv("results/bow_classifier_results/all_urls_revisit.csv", header=FALSE))
classifier <- data.table(read.csv("results/classifier_results/all_urls_revisit.csv", header=FALSE))
names(bow) <- c("batch_num", "Training", "Validation")
names(classifier) <- c("batch_num", "Training", "Validation")
plot_bow <- melt(bow, id.vars='batch_num')
plot_bow$type <- "Bag-of-words"
plot_embed <- melt(classifier, id.vars='batch_num')
plot_embed$type <- "Embedding"
plot_data <- rbind(plot_embed, plot_bow)
plot_data$type <- factor(plot_data$type, levels=c('Embedding', 'Bag-of-words'))
g <- ggplot(data=plot_data, aes(x=batch_num, y=value, color=variable)) + geom_line(size=0.8)
g <- g + labs(x='Batch Number', y='Loss', color='')
g <- g + facet_wrap(~type, scales="free") + theme(legend.position="top")
g

bow_class_vals <- data.table(read.csv("results/bow_classifier_results/test_value.csv"))
bow_class_vals <- bow_class_vals[order(-value)]
bow_class_vals$numeric_value <- as.numeric(bow_class_vals$value)
bow_class_vals$value <- factor(bow_class_vals$value, levels=unique(class_vals$value))
bow_class_vals <- bow_class_vals[, median_value:=median(predicted_value), by=value]
g_class <- ggplot(data=bow_class_vals, aes(x=predicted_value, fill=value)) + geom_histogram(alpha=0.8, color='grey3')
g_class <- g_class + geom_vline(aes(xintercept=numeric_value, group=value), linetype=2, size=0.75)
g_class <- g_class + labs(x="Predicted Value Function", y="", fill="True Value Function")
g_class <- g_class + theme(legend.position="bottom") + guides(fill=guide_legend(nrow=2))
g_class <- g_class + facet_grid(value~., scales = "free_y") + guides(fill=FALSE, color=FALSE)
g_class + scale_x_continuous(limits=c(0, 1))

# BOW feature coefficients
bow_coefs <- data.table(read.csv("results/classifier_results/feature_coefs.csv"))
# feature_coefs <- data.table(read.csv(paste0("results/text_linear_results/", feature_weights_path)))
bow_coefs$coef_mag <- abs(bow_coefs$coef)
bow_coefs <- bow_coefs[order(-coef_mag)]
bow_coefs$words <- factor(bow_coefs$words, levels=unique(bow_coefs$words))

g_coef <- ggplot(data=bow_coefs[1:20], aes(x=words, y=coef, fill=words)) + geom_bar(stat='identity')
g_coef <- g_coef + theme(legend.position='none', axis.text.x=element_text(angle=45, hjust=1))
g_coef <- g_coef + labs(x="", y="Weight")
g_coef
ggsave(filename=paste0("../figures/our_work/", save_coefs_ending), plot=g_coef, width=15, height=10, units="cm")

# Embedding 2 million steps
vals_2m <- data.table(read.csv("results/embedding_buffer_results/test_rep.csv"))
vals_2m <- vals_2m[order(-value)]
vals_2m$value <- factor(vals_2m$value, levels=unique(class_vals$value))
vals_2m <- vals_2m[, median_value:=median(predicted_value), by=value]
g_class <- ggplot(data=vals_2m, aes(x=predicted_value, fill=value)) + geom_histogram(alpha=0.8, color='grey3')
g_class <- g_class + geom_vline(aes(xintercept=median_value, group=value), linetype=2, size=0.75)
g_class <- g_class + labs(x="Predicted Value Function", y="", fill="True Value Function")
g_class <- g_class + theme(legend.position="bottom") + guides(fill=guide_legend(nrow=2))
g_class <- g_class + facet_grid(value~., scales = "free_y") + guides(fill=FALSE, color=FALSE)
g_class + scale_x_continuous(limits=c(0, 1))


##-------------------- Distribution of number of links in the web graph
rl_web_links <- data.table(read.csv("../uk_web_application/building_data/test_results/rl_web_graph_links.csv"))
# rl_web_links <- rl_web_links[num_links<=500]
g_rl_links <- ggplot(data=rl_web_links, aes(x=num_links)) + geom_histogram(color='firebrick', fill='lightblue')
g_rl_links <- g_rl_links + labs(x='Number of Outgoing Links', y='Count')
g_rl_links
summary(rl_web_links$num_links)

