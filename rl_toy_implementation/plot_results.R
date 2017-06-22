library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)
library(scales)

#-------------------- Read random train results data and plot
random_results <- read.csv("results/random_crawler_results_new.csv")
random_results$type <- "Random Crawler"
# dqn_results <- read.csv("results/results_tues_20/dqn_crawler_train_results_50k.csv")
dqn_results <- read.csv("results/dqn_crawler_train_results_new.csv")
dqn_results$type <- "DQN Agent"

df <- rbind(random_results, subset(dqn_results, select=-c(nn_loss)))
df$type <- factor(df$type)

## Reward plot
g_reward <- ggplot(data=df, aes(x=pages_crawled, y=total_reward, color=type)) 
g_reward <- g_reward + geom_line(size=0.9) + labs(x='Pages Crawled', y='Total Reward', color='')
g_reward <- g_reward + theme(legend.position='top')
g_reward <- g_reward + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)

## Terminal states plot
g_terminal <- ggplot(data=df, aes(x=pages_crawled, y=terminal_states, color=type)) 
g_terminal <- g_terminal + geom_line(size=0.9) + labs(x='Pages Crawled', y='Number Terminal States', color='')
g_terminal <- g_terminal + theme(legend.position='top')
g_terminal <- g_terminal + scale_x_continuous(labels=comma) + scale_y_continuous(labels=comma)

## Combine plots
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
g <- grid_arrange_shared_legend(g_reward, g_terminal, nrow = 1, ncol = 2)
g
# ggsave(filename="../figures/our_work/dqn_vs_random.png", plot=g, width = 15, height = 10, units = "cm")


##---------------------- Plot the slope
df$obs <- 1:nrow(df)
df <- df[df$obs %% 500 == 0, ]
df$slope <- c(0, diff(df$total_reward))
df[df$slope<0, ]$slope <- 0
g_slope <- ggplot(data=df, aes(x=pages_crawled, y=slope, color=type)) + geom_line() + facet_grid(type~.)
g_slope <- g_slope + theme(legend.position='top') + labs(x='Pages Crawled', y='Slope')
g_slope


##----------------------- Look at feature coefficients
feature_coefs <- data.table(read.csv("results/feature_coefficients.csv"))
feature_coefs$coef_mag <- abs(feature_coefs$coef)
feature_coefs <- feature_coefs[order(-coef_mag)]
feature_coefs$words <- factor(feature_coefs$words, levels=feature_coefs$words)

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


##--------------------- Find unique fraction of URLs that the crawler got
## DQN crawler
reward_urls <- read.csv("results/reward_pages.csv")
reward_urls <- as.character(reward_urls$rewards_pages)
length(unique(sub("/.*$","", reward_urls)))

## Random crawler
reward_urls <- read.csv("results/random_reward_pages.csv")
reward_urls <- as.character(reward_urls$rewards_pages)
length(unique(sub("/.*$","", reward_urls)))

