list_of_packages <- c("ggplot2", "dplyr", "tidyr", "circlize","arrow", "ggrepel")
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



figures_path <- file.path("../figures/mAP")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

organoid_fs_intra_patient_mAP_file <- file.path(
    root_dir,
    "5.EDA/results/mAP/organoid_fs_intra_patient_mAP.parquet"
)
organoid_fs_inter_patient_mAP_file <- file.path(
    root_dir,
    "5.EDA/results/mAP/sc_fs_inter_patient_mAP.parquet"
)
sc_fs_intra_patient_mAP_file <- file.path(
    root_dir,
    "5.EDA/results/mAP/sc_fs_intra_patient_mAP.parquet"
)
sc_fs_inter_patient_mAP_file <- file.path(
    root_dir,
    "5.EDA/results/mAP/sc_fs_inter_patient_mAP.parquet"
)
organoid_fs_intra_patient_mAP_df <- arrow::read_parquet(
    organoid_fs_intra_patient_mAP_file
)
organoid_fs_inter_patient_mAP_df <- arrow::read_parquet(
    organoid_fs_inter_patient_mAP_file
)
sc_fs_intra_patient_mAP_df <- arrow::read_parquet(
    sc_fs_intra_patient_mAP_file
)
sc_fs_inter_patient_mAP_df <- arrow::read_parquet(
    sc_fs_inter_patient_mAP_file
)

# plot paths
organoid_fs_intra_patient_mAP_plot_path <- file.path(
    figures_path,
    "organoid_fs_intra_patient_mAP.png"
)
organoid_fs_inter_patient_mAP_plot_path <- file.path(
    figures_path,
    "organoid_fs_inter_patient_mAP.png"
)
sc_fs_intra_patient_mAP_plot_path <- file.path(
    figures_path,
    "sc_fs_intra_patient_mAP.png"
)
sc_fs_inter_patient_mAP_plot_path <- file.path(
    figures_path,
    "sc_fs_inter_patient_mAP.png"
)


width <- 12
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
organoid_intra_plot <- (
    ggplot(data = organoid_fs_intra_patient_mAP_df,
    aes(
        x = mean_average_precision,
        y = `-log10(p-value)`,
        color = treatment
    ))
    + geom_point(size = 2, alpha = 0.7)
    + labs(
        x = "Mean Average Precision",
        y = "-log10(p-value)",
        title = "Intra-patient mAP for Organoid FS"
    )
    + geom_hline(yintercept = 1.3, linetype = "dashed", color = "red")  # Adjust threshold as needed
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
    + xlim(0, 1)
    + facet_wrap(~patient, ncol = 4,
    # scales = "free"
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )
)
ggsave(
    filename = organoid_fs_intra_patient_mAP_plot_path,
    plot = organoid_intra_plot,
    width = width,
    height = height,
    dpi = 600
)
organoid_intra_plot

width <- 8
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
organoid_inter_plot <- (
    ggplot(data = organoid_fs_inter_patient_mAP_df,
    aes(
        x = mean_average_precision,
        y = `-log10(p-value)`,
        color = treatment
    ))
    + geom_point(size = 4, alpha = 0.7)
    + labs(
        x = "Mean Average Precision",
        y = "-log10(p-value)",
        title = "Inter-patient mAP for Organoid FS"
    )
    + geom_hline(yintercept = 1.3, linetype = "dashed", color = "red")  # Adjust threshold as needed
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )
    + xlim(0, 1)
)
ggsave(
    filename = organoid_fs_inter_patient_mAP_plot_path,
    plot = organoid_inter_plot,
    width = width,
    height = height,
    dpi = 600
)
organoid_inter_plot

width <- 10
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
sc_intra_plot <- (
    ggplot(data = sc_fs_intra_patient_mAP_df,
    aes(
        x = mean_average_precision,
        y = `-log10(p-value)`,
        color = treatment
    ))
    + geom_point(size = 2, alpha = 0.7)
    + labs(
        x = "Mean Average Precision",
        y = "-log10(p-value)",
        title = "Intra-patient mAP for single cell FS"
    )
    + geom_hline(yintercept = 1.3, linetype = "dashed", color = "red")  # Adjust threshold as needed
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
    + xlim(0, 1)
    + facet_wrap(~patient, ncol = 4,
    # scales = "free"
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 8
    )
)
ggsave(
    filename = sc_fs_intra_patient_mAP_plot_path,
    plot = sc_intra_plot,
    width = width,
    height = height,
    dpi = 600
)
sc_intra_plot

width <- 12
height <- 8
options(repr.plot.width = width, repr.plot.height = height)
sc_inter_plot <- (
    ggplot(data = sc_fs_inter_patient_mAP_df,
    aes(
        x = mean_average_precision,
        y = `-log10(p-value)`,
        color = treatment
    ))
    + geom_point(size = 4, alpha = 0.7)
    + labs(
        x = "Mean Average Precision",
        y = "-log10(p-value)",
        title = "Inter-patient mAP for single cell FS"
    )
    + geom_hline(yintercept = 1.3, linetype = "dashed", color = "red")  # Adjust threshold as needed
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
    + xlim(0, 1)
    + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )

)
ggsave(
    filename = sc_fs_inter_patient_mAP_plot_path,
    plot = sc_inter_plot,
    width = width,
    height = height,
    dpi = 600
)
sc_inter_plot

organoid_fs_intra_patient_mAP_df <- select(organoid_fs_intra_patient_mAP_df,
    c(
        treatment,mean_average_precision, patient
    )
)
colnames(organoid_fs_intra_patient_mAP_df)[2] <- "intra_patient_mAP"


organoid_fs_inter_patient_mAP_df <- select(organoid_fs_inter_patient_mAP_df,
    c(
        treatment, mean_average_precision
    )
)
# rename mean_average_precision to inter_patient_mAP
colnames(organoid_fs_inter_patient_mAP_df)[2] <- "inter_patient_mAP"
# rename mean_average_precision to intra_patient_mAP
# combine intra and inter patient mAP for organoid
# merge the two dataframes on treatment and shuffle
organoid_sc_fs_mAP_df <- merge(
    organoid_fs_intra_patient_mAP_df,
    organoid_fs_inter_patient_mAP_df,
    by = c("treatment"),
    all = TRUE
)

organoid_sc_fs_mAP_df$intra_to_inter_ratio <- (
    organoid_sc_fs_mAP_df$intra_patient_mAP /
    organoid_sc_fs_mAP_df$inter_patient_mAP
)
head(organoid_sc_fs_mAP_df)

width <- 8
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
ratio_plot <- (
    ggplot(
        data = organoid_sc_fs_mAP_df,
        aes(
            x = treatment,
            y = intra_to_inter_ratio,
            fill = treatment
        )
    )
    + geom_bar(stat = "identity", position = "dodge")
    + labs(
        x = "Treatment",
        y = "Intra to Inter Patient mAP Ratio",
        title = "Intra to Inter Patient mAP Ratio for Organoid and Single Cell FS"
    )
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.text.x = element_text(angle = 90, hjust = 1, size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "none",
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
    )
    + geom_hline(
        yintercept = 1,
        linetype = "dashed",
        color = "red"
    )  # Adjust threshold as needed

)
ggsave(
    filename = file.path(figures_path, "intra_to_inter_patient_mAP_ratio_bar.png"),
    plot = ratio_plot,
    width = width,
    height = height,
    dpi = 600
)
ratio_plot

width <- 10
height <- 10
options(repr.plot.width = width, repr.plot.height = height)
plot <- (
    ggplot(
        data = organoid_sc_fs_mAP_df,
        aes(
            x = inter_patient_mAP,
            y = intra_patient_mAP,
            color = treatment,
            label = treatment
                    )
    )
    + geom_point(size = 2, alpha = 0.7)
    # + geom_boxplot(
    #     aes(fill = treatment),
    #     alpha = 0.4,
    #     size = 5,
    #     outlier.shape = NA,
    #     jitter = TRUE,
    #     position = position_dodge(width = 0.75)
    # )
    + labs(
        x = "Inter-patient mAP",
        y = "Intra-patient mAP",
        title = "Inter vs Intra Patient mAP for Organoid FS"
    )
    # + ylim(0, 1)
    # + xlim(0, 1)
    + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red")  # Diagonal line
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )
    + xlim(0, 1)
)
ggsave(
    filename = file.path(figures_path, "organoid_fs_mAP_inter_vs_intra.png"),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 600
)
plot

organoid_fs_inter_patient_distance_metrics_path <- file.path(
    root_dir,
    "5.EDA/results/distance_metrics/organoid_fs_inter_patient_distance_metrics.parquet"
)
organoid_fs_intra_patient_distance_metrics_path <- file.path(
    root_dir,
    "5.EDA/results/distance_metrics/organoid_fs_intra_patient_distance_metrics.parquet"
)
single_cell_fs_inter_patient_distance_metrics_path <- file.path(
    root_dir,
    "5.EDA/results/distance_metrics/sc_fs_inter_patient_distance_metrics.parquet"
)
single_cell_fs_intra_patient_distance_metrics_path <- file.path(
    root_dir,
    "5.EDA/results/distance_metrics/sc_fs_intra_patient_distance_metrics.parquet"
)

figures_path <- file.path(
    root_dir,
    "5.EDA/figures/distance_metrics/"
)
if (!dir.exists(figures_path)) {
    dir.create(figures_path, recursive = TRUE)
}

organoid_fs_inter_patient_distance_metrics_df <- arrow::read_parquet(
    organoid_fs_inter_patient_distance_metrics_path
)
organoid_fs_intra_patient_distance_metrics_df <- arrow::read_parquet(
    organoid_fs_intra_patient_distance_metrics_path
)
single_cell_fs_inter_patient_distance_metrics_df <- arrow::read_parquet(
    single_cell_fs_inter_patient_distance_metrics_path
)
single_cell_fs_intra_patient_distance_metrics_df <- arrow::read_parquet(
    single_cell_fs_intra_patient_distance_metrics_path
)

head(organoid_fs_inter_patient_distance_metrics_df)

width <- 10
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
plot <- (ggplot(
    data = organoid_fs_inter_patient_distance_metrics_df,
    aes(x = euclidean_distance_mean, y = euclidean_distance_std, fill = treatment, label = treatment)

)
    + geom_point(size = 3, shape = 21, alpha = 0.7)
    + labs(
        x = "Mean Euclidean Distance",
        y = "Standard Deviation of Euclidean Distance",
        title = "Inter-patient Distance Metrics for Organoid Consensus"
    )
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )
)
ggsave(
    filename = file.path(figures_path, "organoid_fs_inter_patient_distance_metrics.png"),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 600
)
plot

width <- 10
height <- 7
options(repr.plot.width = width, repr.plot.height = height)
plot <- (ggplot(
    data = organoid_fs_intra_patient_distance_metrics_df,
    aes(x = euclidean_distance_mean, y = euclidean_distance_std, fill = treatment, label = treatment)

)
    + geom_point(size = 3, shape = 21, alpha = 0.7)
    + labs(
        x = "Mean Euclidean Distance",
        y = "Standard Deviation of Euclidean Distance",
        title = "Intra-patient Distance Metrics for Organoid Consensus"
    )
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )
    + facet_wrap(~patient, ncol = 4,
        scales = "free"
    )
    + scale_x_continuous(breaks = scales::pretty_breaks(n = 4))
    + theme(axis.text.x = element_text(size = 14, angle = 45, hjust = 1))
)
ggsave(
    filename = file.path(figures_path, "organoid_fs_intra_patient_distance_metrics.png"),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 600
)
plot

width <- 10
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
plot <- (ggplot(
    data = single_cell_fs_inter_patient_distance_metrics_df,
    aes(x = euclidean_distance_mean, y = euclidean_distance_std, fill = treatment, label = treatment)

)
    + geom_point(size = 3, shape = 21, alpha = 0.7)
    + labs(
        x = "Mean Euclidean Distance",
        y = "Standard Deviation of Euclidean Distance",
        title = "Inter-patient Distance Metrics for Single Cell Consensus"
    )
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )

)
ggsave(
    filename = file.path(figures_path, "single_cell_fs_inter_patient_distance_metrics.png"),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 600
)
plot

width <- 10
height <- 7
options(repr.plot.width = width, repr.plot.height = height)
plot <- (ggplot(
    data = single_cell_fs_intra_patient_distance_metrics_df,
    aes(x = euclidean_distance_mean, y = euclidean_distance_std, fill = treatment, label = treatment)

)
    + geom_point(size = 3, shape = 21, alpha = 0.7)
    + labs(
        x = "Mean Euclidean Distance",
        y = "Standard Deviation of Euclidean Distance",
        title = "Intra-patient Distance Metrics for Single Cell Consensus"
    )
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16)
    )
        + geom_text_repel(
        aes(
            label = treatment
        ),
        size = 4,
        nudge_y = 0.1,
        show.legend = FALSE,
        segment.color = "grey50",
        segment.size = 0.2,
        box.padding = 0.5,
        point.padding = 0.5,
        max.overlaps = 10
    )
    + facet_wrap(~patient, ncol = 4,
        scales = "free"
    )
    + scale_x_continuous(breaks = scales::pretty_breaks(n = 4))
    + theme(axis.text.x = element_text(size = 14, angle = 45, hjust = 1))
)
ggsave(
    filename = file.path(figures_path, "single_cell_fs_intra_patient_distance_metrics.png"),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 600
)
plot
