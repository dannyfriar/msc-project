library(data.table)
library(ggplot2)
library(scales)

prop <- read.csv('../data/proportion_companies.csv', header=TRUE)
prop <- prop[prop$is_bad_url == 0, ]

# Plot the percentage of company names
g1 <- ggplot(data = prop, aes(x=name_prop)) + geom_histogram(color='firebrick', fill='lightblue', bins=20)
g1 <- g1 + labs(x = '% Company Names in Page', y = "") + scale_x_continuous(labels = percent, breaks=seq(0, 0.01, 0.002))
ggsave(file="../figures/our_work/name_prop.png", g1)
g1

# Plot the percentage of keywords
g2 <- ggplot(data = prop, aes(x=vert_desc_prop)) + geom_histogram(color='firebrick', fill='lightblue', bins=20)
g2 <- g2 + labs(x = '% Vertical Keywords in Page', y = "") + scale_x_continuous(labels = percent)
ggsave(file="../figures/our_work/prop_hist.png", g2)
g2

prop.filtered <- prop[prop$vert_desc_prop <= 0.1 & prop$name_prop <= 0.005, ]
g3 <- ggplot(data=prop.filtered, aes(x=vert_desc_prop, y=name_prop))
g3 <- g3 + geom_point(shape=16, size=2, alpha=.7, show.legend=FALSE, color='firebrick')
g3 <- g3 + labs(x='% Vertical Keywords in Page', y='% Company Names in Page')
g3 <- g3 + scale_x_continuous(labels = percent) + scale_y_continuous(labels = percent) + theme_minimal()
ggsave(file="../figures/our_work/comp_scatter.png", g3)
g3

cor(prop$vert_desc_prop, prop$name_prop)


#------------------------------------
# Load check rewards data
check_df <- read.csv('results/check_rewards.csv')
# check_df$name <- as.character(check_df$name)
# check_df$name_length <- nchar(check_df$name)
# check_df <- check_df[order(check_df$name_length), ]

check_df <- check_df[check_df$count > 0, ]
check_df <- check_df[order(-check_df$count), ]


# Load vert frequency data
vert_df <- read.csv('results/vert_freq.csv')
View(vert_df)


# Load crawled pages
crawled_pages <- read.csv('results/crawled_pages.csv')
View(crawled_pages)






