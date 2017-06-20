library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)

# Read random train results data and plot
random_results <- read.csv("results/results_tues_20/random_crawler_train_results.csv")
random_results$type <- "Random Crawler"
dqn_results <- read.csv("results/results_tues_20/dqn_crawler_train_results.csv")
dqn_results$type <- "DQN Agent"

df <- rbind(random_results, subset(dqn_results, select=-c(nn_loss)))
df$type <- factor(df$type)

## Reward plot
g_reward <- ggplot(data=df, aes(x=pages_crawled, y=total_reward, color=type)) 
g_reward <- g_reward + geom_line(size=0.9) + labs(x='Pages Crawled', y='Total Reward', color='')
g_reward <- g_reward + theme(legend.position='top')

## Terminal states plot
g_terminal <- ggplot(data=df, aes(x=pages_crawled, y=terminal_states, color=type)) 
g_terminal <- g_terminal + geom_line(size=0.9) + labs(x='Pages Crawled', y='Number Terminal States', color='')
g_terminal <- g_terminal + theme(legend.position='top')

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
print(g)
ggsave(filename="../figures/our_work/dqn_vs_random.png", plot=g, width = 15, height = 10, units = "cm")

