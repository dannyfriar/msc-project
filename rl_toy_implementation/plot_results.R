library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)

# Read random train results data and plot
random_results <- read.csv("results/random_crawler_train_results.csv")
random_results$type <- "Random Crawler"
dqn_results <- read.csv("results/dqn_crawler_train_results.csv")
dqn_results$type <- "Initialized DQN Agent"

df <- rbind(random_results, subset(dqn_results, select=-c(nn_loss)))
df$type <- factor(df$type)

## Reward plot
g_reward <- ggplot(data=df, aes(x=pages_crawled, y=total_reward, color=type)) 
g_reward <- g_reward + geom_line() + labs(x='Pages Crawled', y='Total Reward', color='')
g_reward <- g_reward + theme(legend.position='top')

## Terminal states plot
g_terminal <- ggplot(data=df, aes(x=pages_crawled, y=terminal_states, color=type)) 
g_terminal <- g_terminal + geom_line() + labs(x='Pages Crawled', y='Number Terminal States', color='')
g_terminal <- g_terminal + theme(legend.position='top')

grid.arrange(g_reward, g_terminal, ncol=1)
