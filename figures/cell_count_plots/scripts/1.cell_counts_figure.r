list_of_packages <- c("ggplot2", "dplyr", "tidyr", "circlize", "platetools")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                package,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
        )
    )
}
# Get the current working directory and find Git root
find_git_root <- function() {
    # Get current working directory
    cwd <- getwd()

    # Check if current directory has .git
    if (dir.exists(file.path(cwd, ".git"))) {
        return(cwd)
    }

    # If not, search parent directories
    current_path <- cwd
    while (dirname(current_path) != current_path) {  # While not at root
        parent_path <- dirname(current_path)
        if (dir.exists(file.path(parent_path, ".git"))) {
            return(parent_path)
        }
        current_path <- parent_path
    }

    # If no Git root found, stop with error
    stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()
cat("Git root directory:", root_dir, "\n")
figures_path <- file.path(root_dir, "figures/cell_count_plots/figures")
if (!dir.exists(figures_path)) {
    dir.create(figures_path, recursive = TRUE)
}

cell_counts_theme <- theme(
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18),
        plot.title = element_text(size = 20, hjust = 0.5),
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
        strip.text = element_text(size = 16),
        legend.position = "none"
    )

sc_organoid_profile_path <- file.path(
    root_dir,
    "figures/cell_count_plots/results/sc_and_organoid_counts.parquet"
)
df <- arrow::read_parquet(sc_organoid_profile_path)
head(df)

df$treatment <- factor(
    df$treatment,
    levels = c(
        'DMSO',
        'Staurosporine',
        'Binimetinib',
        'Cabozantinib',
        'Copanlisib',
        'Digoxin',
        'Everolimus',
        'Fimepinostat',
        'Imatinib',
        'Ketotifen',
        'Linsitinib',
        'Mirdametinib',
        'Nilotinib',
        'Onalespib',
        'Rapamycin',
        'Selumetinib',
        'Trametinib',
        'ARV-825',
        'Panobinostat',
        'Sapanisertib',
        'Trabectedin',
        'Vistusertib'
    )
)


width <- 14
height <- 14
options(repr.plot.width = width, repr.plot.height = height)
single_cell_counts_plot <- (
    ggplot(
        data = df,
        aes(x = patient, y = single_cell_count, fill = patient)
    )
    + geom_boxplot(alpha = 0.5)
    + geom_jitter(
        aes(color = patient),
        size = 1,
        alpha = 0.5,
        width = 0.2,
        height = 0

    )
    + labs(
        title = "Single-cell counts per treatment",
        x = "Treatment",
        y = "Single-cell count"
    )
    + theme_bw()
    + cell_counts_theme
    + facet_wrap(~ treatment, ncol = 5)
)
ggsave(
    plot = single_cell_counts_plot,
    filename = file.path(figures_path, "single_cell_counts_per_treatment.png"),
    width = width,
    height = height
)
single_cell_counts_plot

width <- 14
height <- 14
options(repr.plot.width = width, repr.plot.height = height)
organoid_cell_counts_plot <- (
    ggplot(
        data = df,
        aes(x = patient, y = organoid_count, fill = patient)
    )
    + geom_boxplot(alpha = 0.5)
    + geom_jitter(
        aes(color = patient),
        size = 1,
        alpha = 0.5,
        width = 0.2,
        height = 0

    )
    + labs(
        title = "Organoid counts per treatment",
        x = "Treatment",
        y = "Organoid count"
    )
    + theme_bw()
    + cell_counts_theme
    + facet_wrap(~ treatment, ncol = 5)
)
ggsave(
    plot = organoid_cell_counts_plot,
    filename = file.path(figures_path, "organoid_counts_per_treatment.png"),
    width = width,
    height = height
)
organoid_cell_counts_plot

width <- 14
height <- 14
options(repr.plot.width = width, repr.plot.height = height)
cell_count_volume_ratio_plot <- (
    ggplot(
        data = df,
        aes(x = patient, y = cell_density)
    )
    + geom_boxplot(aes(fill = patient), alpha = 0.5)
    + geom_jitter(
        aes(color = patient),
        size = 1,
        alpha = 0.5,
        width = 0.2,
        height = 0    )
    + labs(
        title = "Cell Density per Organoid",
        x = "Treatment",
        y = "Cell Density (cells/um^3)"
    )
    + theme_bw()
    + ylim(c(0, quantile(df$cell_density, 0.99, na.rm = TRUE)))  # Fixed: added na.rm = TRUE
    + cell_counts_theme

    + facet_wrap(~ treatment, ncol = 5)
)
ggsave(
    plot = cell_count_volume_ratio_plot,
    filename = file.path(figures_path, "cell_density_per_organoid.png"),
    width = width,
    height = height
)
cell_count_volume_ratio_plot
