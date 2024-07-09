Sys.setlocale("LC_TIME", "en_US.UTF-8")

# Load required libraries
library(ggplot2)
#library(hrbrthemes)
library(optparse)
library(dplyr)

# Set up command-line options
option_list <- list(
  make_option(c("-d", "--dir"), type = "character", default = "some dir", help = "Directory containing date folders", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", default = "deeplili_stats.png", help = "Output filename for the histogram", metavar = "character")
)

# Parse command-line arguments
args <- parse_args(OptionParser(option_list = option_list))
deeplili_dir <- args$dir
output_file <- args$output

# List directories in the specified path
directories <- list.dirs(path = deeplili_dir, full.names = FALSE, recursive = FALSE)

# Convert directory names to Date objects assuming they are in 'YYYY-MM-DD' format
date_directories <- as.Date(directories, format = "%Y-%m-%d")

# Create a data frame for plotting
date_data <- data.frame(dates = date_directories)
current_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")


# Assuming date_data is your dataframe and it contains a column named 'dates'
date_data <- date_data %>%
  mutate(day_of_week = weekdays(dates),
         weekend = ifelse(day_of_week %in% c("Sunday", "Saturday"), "Weekend", "Weekday"))

print(date_data)

# Create and print the histogram
ggplot(date_data, aes(y = dates)) +
  geom_histogram(binwidth = 1, color = "black", aes(fill = weekend, x = after_stat(count))) +
  scale_fill_manual(values = c("Weekend" = "orange", "Weekday" = "green")) +
  geom_text(stat = 'count', aes(label = after_stat(count)), hjust = -0.1) +  # Add count labels
  ggdark::dark_theme_classic() +
  labs(title = "DeepLili MMAMM 2024", subtitle = paste("Per day activity on", current_time), y = "Date", x = "Frequency")





# Create and print the histogram
#ggplot(date_data, aes(y = dates)) +
#  geom_histogram(binwidth = 1, color = "black", fill = "green", aes(x = after_stat(count) )) +
#  geom_text(stat = 'count', aes(label = after_stat(count)), hjust = -0.1) +  # Add count labels
#  ggdark::dark_theme_classic()+
#  labs(title = "DeepLili MMAMM 2024", subtitle = paste("Per day activity on",current_time), y = "Date", x = "Frequency")
#  scale_y_date(date_breaks = "1 day", date_labels = "%Y-%m-%d") 
# scale_y_discrete(breaks = date_data$dates)  # Show all dates




# Save the plot
ggsave(output_file, width = 14, height = 5, units = 'in')

