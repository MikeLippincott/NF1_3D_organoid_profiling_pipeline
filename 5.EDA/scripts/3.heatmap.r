list_of_packages <- c(
       "ggplot2",
       "dplyr",
       "tidyr",
       "ComplexHeatmap",
       "tibble",
       "RColorBrewer",
       "scales",
       "circlize",
       "argparse"
)
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")

if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
    BiocManager::install("ComplexHeatmap")
}
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

# set custom colors for each MOA
custom_MOA_palette <- c(
    'BRD4 inhibitor' = "#93152A",  # Dark red
    'receptor tyrosine kinase inhibitor' = "#BA3924",  # Red
    'tyrosine kinase inhibitor' = "#D08543",  # Orange
    'MEK1/2 inhibitor' = "#A1961A",  # Yellow-green/olive

    'IGF-1R inhibitor' = "#9FC62A",  # Yellow-green
    'mTOR inhibitor' = "#1FAD23",  # Green
    'PI3K inhibitor' = "#32D06A",  # Light green
    'PI3K and HDAC inhibitor' = "#15937C",  # Teal/dark green
    'HDAC inhibitor' = "#24A5BA",  # Light blue/cyan

    'Apoptosis induction' = "#438CD0",  # Medium blue
    'DNA binding' = "#1A24A1",  # Dark blue
    'HSP90 inhibitor' = "#532AC6",  # Blue-purple

    'histamine H1 receptor antagonist' = "#AD1FA6",  # Purple/magenta
    'Na+/K+ pump inhibitor' = "#D03294",  # Pink/magenta

    'Control' = "#444444"  # Gray
)

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

sc_consensus_df <- arrow::read_parquet(file.path(root_dir,"data/all_patient_profiles/sc_consensus_profiles.parquet"))
sc_fs_df <- arrow::read_parquet(file.path(root_dir,"data/all_patient_profiles/sc_fs_profiles.parquet"))
# drop the therapeutic category column
sc_consensus_df <- sc_consensus_df %>%
  select(-`Therapeutic Categories`)
sc_fs_df <- sc_fs_df %>%
  select(-`Therapeutic Categories`)

organoid_consensus_df <- arrow::read_parquet(file.path(root_dir,"data/all_patient_profiles/organoid_consensus_profiles.parquet"))
organoid_fs_df <- arrow::read_parquet(file.path(root_dir,"data/all_patient_profiles/organoid_fs_profiles.parquet"))
# drop the therapeutic category column
organoid_consensus_df <- organoid_consensus_df %>%
  select(-`Therapeutic Categories`)
organoid_fs_df <- organoid_fs_df %>%
  select(-`Therapeutic Categories`)


sc_consensus_heatmap_file_path <- file.path(
    root_dir,
    paste0("5.EDA/figures/heatmaps/sc_consensus_heatmap.png")
)
sc_fs_heatmap_file_path <- file.path(
    root_dir,
    paste0("5.EDA/figures/heatmaps/sc_fs_heatmap.png")
)
organoid_consensus_heatmap_file_path <- file.path(
    root_dir,
    paste0("5.EDA/figures/heatmaps/organoid_consensus_heatmap.png")
)
organoid_fs_heatmap_file_path <- file.path(
    root_dir,
    paste0("5.EDA/figures/heatmaps/organoid_fs_heatmap.png")
)


if (!dir.exists(file.path(root_dir,paste0("5.EDA/figures/heatmaps/")))) {
    dir.create(file.path(root_dir,paste0("5.EDA/figures/heatmaps/")), recursive = TRUE)
}

patient_colors <- rainbow(length(unique(organoid_fs_df$patient)))
names(patient_colors) <- unique(organoid_fs_df$patient)


head(sc_consensus_df)

# drop columns that contain neighbors
sc_consensus_df <- sc_consensus_df %>%
  select(-contains("Neighbors"))

column_anno <- HeatmapAnnotation(
    Target = sc_consensus_df$Target,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    col = list(
            Target = custom_MOA_palette
        )
)
patient_columns_anno <- HeatmapAnnotation(
    patient = sc_consensus_df$patient,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16)))

)

column_annotations = c(column_anno, patient_columns_anno)

# get the list of features
features <- colnames(sc_consensus_df)
features <- features[!features %in% c("treatment", "Target", "Class", "single_cell_count","patient","dose","unit")]
features <- as.data.frame(features)

rownames(features) <- features$features
# split the features by _ into multiple columns
features <- features %>%
    separate(features, into = c("Feature Type", "Compartment", "Channel", "Measurement"), sep = "_", extra = "merge", fill = "right")
# if Feature type is AreaSizeShape then shift the Channel to the Measurement column and set Channel to NA
features <- features %>%
    mutate(
        Measurement = ifelse(`Feature Type` == "Area.Size.Shape", Channel, Measurement)
    )
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Area.Size.Shape", "None", Channel)
    )

# select the first channel for colocalization features channels are split by .
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Colocalization",
                         sub("\\..*", "", Channel),
                         Channel)
    )


# sort by feature type
features <- features %>%
    arrange(`Feature Type`, Compartment, Channel, Measurement)

# compartment row annotation
row_compartment = rowAnnotation(
    Object = features$Compartment,
        show_legend = TRUE,
    # change the legend titles
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    annotation_name_side = "bottom",
    annotation_name_gp = gpar(fontsize = 16),
    # color
    col = list(
        Object = c(
            "Cell" = "#B000B0",
            "Cytoplasm" = "#00D55B",
            "Nuclei" = "#0000AB"
            )
    )
)
row_measurement = rowAnnotation(
    FeatureType = features$`Feature Type`,
           annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    annotation_name_side = "bottom",
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
            FeatureType = c(
            "Area.Size.Shape" = brewer.pal(8, "Paired")[1],
            "Colocalization" = brewer.pal(8, "Paired")[2],
            "Granularity" = brewer.pal(8, "Paired")[3],
            "Intensity" = brewer.pal(8, "Paired")[4],
            "Texture" = brewer.pal(8, "Paired")[8]
        )
    ),
    show_legend = TRUE
)
row_channel = rowAnnotation(
    Channel = features$Channel,
        annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        # make annotation bar text bigger
        legend = gpar(fontsize = 16),
        annotation_name = gpar(fontsize = 16),
        # legend_height = unit(20, "cm"),
        legend_width = unit(1, "cm"),
        # make legend taller
        # legend_height = unit(10, "cm"),
        legend_width = unit(1, "cm"),
        legend_key = gpar(fontsize = 16)
        )
    ),



    annotation_name_side = "bottom",
    # make font size bigger
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
    Channel = c(
            "DNA" = "#0000AB",
            "AGP" = "#b1001a",
            "Mito" = "#B000B0",
            "ER" = "#00D55B",
            "BF" = "#FFFF00",
            "None" = "#B09FB0")
    )
)
row_annotations = c(row_compartment, row_measurement, row_channel)

mat <- sc_consensus_df %>%
  select(-treatment, -Class, -Target,-patient,-dose,-unit) %>%

  as.matrix()
mat <- t(mat)
colnames(mat) <- sc_consensus_df$treatment
dim(mat)

width <- 20
height <- 15
options(repr.plot.width = width, repr.plot.height = height)
heatmap_plot <- Heatmap(
        mat,
        # col = col_fun,
        show_row_names = FALSE,
        # cluster_columns = FALSE,
        show_column_names = FALSE,
        row_dend_reorder = TRUE, # reorder rows based on dendrogram
        column_dend_reorder = TRUE, # reorder columns based on dendrogram
        show_row_dend = FALSE,
        show_column_dend = FALSE,

        column_names_gp = gpar(fontsize = 16), # Column name label formatting
        row_names_gp = gpar(fontsize = 14),

        # show_heatmap_legend = FALSE,
        heatmap_legend_param = list(
                    title = "Norm\nValue",
                    title_position = "topcenter",
                    title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
                    labels_gp = gpar(fontsize = 16),
                    # legend_height = unit(2, "cm"),
                    legend_width = unit(3, "cm"),
                    annotation_legend_side = "bottom"
                    ),

        row_dend_width = unit(1, "cm"),
        column_dend_height = unit(1, "cm"),

        # column_title = paste0("Dose: ", dose," uM"),
        right_annotation = row_annotations,
        top_annotation = column_annotations,
        # adjust the title position and size
        column_title_gp = gpar(fontsize = 16, fontface = "bold", hjust = 0.5),

    )

png(sc_consensus_heatmap_file_path, width = width, height = height, units = "in", res = 600)
# save as a PNG
draw(heatmap_plot, merge_legend = TRUE, heatmap_legend_side = "right")
dev.off()
heatmap_plot

head(sc_fs_df)

sc_fs_df <- sc_fs_df %>%
  group_by(patient, image_set) %>%
  mutate(Area.Size.Shape_Cell_CENTER.Z = (
    Area.Size.Shape_Cell_CENTER.Z - min(Area.Size.Shape_Cell_CENTER.Z, na.rm = TRUE)
  ) / (
    max(Area.Size.Shape_Cell_CENTER.Z, na.rm = TRUE) - min(Area.Size.Shape_Cell_CENTER.Z, na.rm = TRUE)
  )) %>%
  ungroup()

# drop columns that contain neighbors
sc_fs_df <- sc_fs_df %>%
  select(-contains("Neighbors"))

column_anno <- HeatmapAnnotation(
    Target = sc_fs_df$Target,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    col = list(
            Target = custom_MOA_palette
        )
)

patient_colors <- rainbow(length(unique(sc_fs_df$patient)))
names(patient_colors) <- unique(sc_fs_df$patient)

patient_columns_anno <- HeatmapAnnotation(
    Patient = sc_fs_df$patient,
    col = list(Patient = patient_colors),
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16)))
)

depth_anno <- HeatmapAnnotation(
    Depth = sc_fs_df$Area.Size.Shape_Cell_CENTER.Z,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16)))
)

column_annotations = c(column_anno, patient_columns_anno, depth_anno)


# get the list of features
features <- colnames(sc_fs_df)
features <- features[!features %in% c("treatment", "Target", "Class", "single_cell_count", "patient",
         "object_id",
         "unit",
         "dose",
         "image_set",
         "Well",
         "parent_organoid",
         "Area.Size.Shape_Cell_CENTER.X",
         "Area.Size.Shape_Cell_CENTER.Y",
         "Area.Size.Shape_Cell_CENTER.Z")]
features <- as.data.frame(features)
rownames(features) <- features$features
# split the features by _ into multiple columns
features <- features %>%
    separate(features, into = c("Feature Type", "Compartment", "Channel", "Measurement"), sep = "_", extra = "merge", fill = "right")
# if Feature type is AreaSizeShape then shift the Channel to the Measurement column and set Channel to NA
features <- features %>%
    mutate(
        Measurement = ifelse(`Feature Type` == "Area.Size.Shape", Channel, Measurement)
    )
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Area.Size.Shape", "None", Channel)
    )

# select the first channel for colocalization features channels are split by .
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Colocalization",
                         sub("\\..*", "", Channel),
                         Channel)
    )


# sort by feature type
features <- features %>%
    arrange(`Feature Type`, Compartment, Channel, Measurement)

# compartment row annotation
row_compartment = rowAnnotation(
    Object = features$Compartment,
        show_legend = TRUE,
    # change the legend titles
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    annotation_name_side = "bottom",
    annotation_name_gp = gpar(fontsize = 16),
    # color
    col = list(
        Object = c(
            "Cell" = "#B000B0",
            "Cytoplasm" = "#00D55B",
            "Nuclei" = "#0000AB"
            )
    )
)
row_measurement = rowAnnotation(
    FeatureType = features$`Feature Type`,
           annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    annotation_name_side = "bottom",
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
            FeatureType = c(
            "Area.Size.Shape" = brewer.pal(8, "Paired")[1],
            "Colocalization" = brewer.pal(8, "Paired")[2],
            "Granularity" = brewer.pal(8, "Paired")[3],
            "Intensity" = brewer.pal(8, "Paired")[4],
            "Texture" = brewer.pal(8, "Paired")[8]
        )
    ),
    show_legend = TRUE
)
row_channel = rowAnnotation(
    Channel = features$Channel,
        annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        # make annotation bar text bigger
        legend = gpar(fontsize = 16),
        annotation_name = gpar(fontsize = 16),
        # legend_height = unit(20, "cm"),
        legend_width = unit(1, "cm"),
        # make legend taller
        # legend_height = unit(10, "cm"),
        legend_width = unit(1, "cm"),
        legend_key = gpar(fontsize = 16)
        )
    ),



    annotation_name_side = "bottom",
    # make font size bigger
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
    Channel = c(
            "DNA" = "#0000AB",
            "AGP" = "#b1001a",
            "Mito" = "#B000B0",
            "ER" = "#00D55B",
            "BF" = "#FFFF00",
            "None" = "#B09FB0")
    )
)
row_annotations = c(row_compartment, row_measurement, row_channel)

mat <- sc_fs_df %>%
  select(-treatment, -Class, -Target, -patient,
        -object_id,
         -unit,
         -dose,
         -image_set,
         -Well,
         -parent_organoid,
         -Area.Size.Shape_Cell_CENTER.X,
         -Area.Size.Shape_Cell_CENTER.Y,
         -Area.Size.Shape_Cell_CENTER.Z
         ) %>%

  as.matrix()
mat <- t(mat)
colnames(mat) <- sc_fs_df$treatment
dim(mat)

width <- 20
height <- 15
options(repr.plot.width = width, repr.plot.height = height)
heatmap_plot <- Heatmap(
        mat,
        # col = col_fun,
        show_row_names = FALSE,
        # cluster_columns = FALSE,
        show_column_names = FALSE,
        row_dend_reorder = TRUE, # reorder rows based on dendrogram
        column_dend_reorder = TRUE, # reorder columns based on dendrogram
        show_row_dend = FALSE,
        show_column_dend = FALSE,

        column_names_gp = gpar(fontsize = 16), # Column name label formatting
        row_names_gp = gpar(fontsize = 14),

        # show_heatmap_legend = FALSE,
        heatmap_legend_param = list(
                    title = "Norm\nValue",
                    title_position = "topcenter",
                    title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
                    labels_gp = gpar(fontsize = 16),
                    # legend_height = unit(2, "cm"),
                    legend_width = unit(3, "cm"),
                    annotation_legend_side = "bottom"
                    ),

        row_dend_width = unit(4, "cm"),
        column_dend_height = unit(4, "cm"),
        # column_title = paste0("Dose: ", dose," uM"),
        right_annotation = row_annotations,
        top_annotation = column_annotations,
        # adjust the title position and size
        column_title_gp = gpar(fontsize = 16, fontface = "bold", hjust = 0.5),

    )
png(sc_fs_heatmap_file_path, width = width, height = height, units = "in", res = 600)
# save as a PNG
draw(heatmap_plot, merge_legend = TRUE, heatmap_legend_side = "right")
dev.off()
heatmap_plot

head(organoid_consensus_df)

# drop columns that contain neighbors
organoid_consensus_df <- organoid_consensus_df %>%
  select(-contains("Neighbors"))

column_anno <- HeatmapAnnotation(
    Target = organoid_consensus_df$Target,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    col = list(
            Target = custom_MOA_palette
        )
)
patient_columns_anno <- HeatmapAnnotation(
    patient = organoid_consensus_df$patient,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16)))

)

column_annotations = c(column_anno, patient_columns_anno)

head(organoid_consensus_df)

# get the list of features
features <- colnames(organoid_consensus_df)
features <- features[!features %in% c("treatment", "Target", "Class", "single_cell_count","patient","dose","unit")]
features <- as.data.frame(features)

rownames(features) <- features$features
# split the features by _ into multiple columns
features <- features %>%
    separate(features, into = c("Feature Type", "Compartment", "Channel", "Measurement"), sep = "_", extra = "merge", fill = "right")
# if Feature type is AreaSizeShape then shift the Channel to the Measurement column and set Channel to NA
features <- features %>%
    mutate(
        Measurement = ifelse(`Feature Type` == "Area.Size.Shape", Channel, Measurement)
    )
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Area.Size.Shape", "None", Channel)
    )

# select the first channel for colocalization features channels are split by .
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Colocalization",
                         sub("\\..*", "", Channel),
                         Channel)
    )


# sort by feature type
features <- features %>%
    arrange(`Feature Type`, Compartment, Channel, Measurement)

row_measurement = rowAnnotation(
    FeatureType = features$`Feature Type`,
           annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    annotation_name_side = "bottom",
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
            FeatureType = c(
            "Area.Size.Shape" = brewer.pal(8, "Paired")[1],
            "Colocalization" = brewer.pal(8, "Paired")[2],
            "Granularity" = brewer.pal(8, "Paired")[3],
            "Intensity" = brewer.pal(8, "Paired")[4],
            "Texture" = brewer.pal(8, "Paired")[8]
        )
    ),
    show_legend = TRUE
)
row_channel = rowAnnotation(
    Channel = features$Channel,
        annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        # make annotation bar text bigger
        legend = gpar(fontsize = 16),
        annotation_name = gpar(fontsize = 16),
        # legend_height = unit(20, "cm"),
        legend_width = unit(1, "cm"),
        # make legend taller
        # legend_height = unit(10, "cm"),
        legend_width = unit(1, "cm"),
        legend_key = gpar(fontsize = 16)
        )
    ),



    annotation_name_side = "bottom",
    # make font size bigger
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
    Channel = c(
            "DNA" = "#0000AB",
            "AGP" = "#b1001a",
            "Mito" = "#B000B0",
            "ER" = "#00D55B",
            "BF" = "#FFFF00",
            "None" = "#B09FB0")
    )
)
row_annotations = c(row_measurement, row_channel)

mat <- organoid_consensus_df %>%
  select(-treatment, -Class, -Target,-patient,-dose,-unit) %>%

  as.matrix()
mat <- t(mat)
colnames(mat) <- organoid_consensus_df$treatment
dim(mat)

width <- 20
height <- 15
options(repr.plot.width = width, repr.plot.height = height)
heatmap_plot <- Heatmap(
        mat,
        # col = col_fun,
        show_row_names = FALSE,
        # cluster_columns = FALSE,
        show_column_names = FALSE,
        row_dend_reorder = TRUE, # reorder rows based on dendrogram
        column_dend_reorder = TRUE, # reorder columns based on dendrogram
        show_row_dend = FALSE,
        show_column_dend = FALSE,

        column_names_gp = gpar(fontsize = 16), # Column name label formatting
        row_names_gp = gpar(fontsize = 14),

        # show_heatmap_legend = FALSE,
        heatmap_legend_param = list(
                    title = "Norm\nValue",
                    title_position = "topcenter",
                    title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
                    labels_gp = gpar(fontsize = 16),
                    # legend_height = unit(2, "cm"),
                    legend_width = unit(3, "cm"),
                    annotation_legend_side = "bottom"
                    ),

        row_dend_width = unit(1, "cm"),
        column_dend_height = unit(1, "cm"),

        # column_title = paste0("Dose: ", dose," uM"),
        right_annotation = row_annotations,
        top_annotation = column_annotations,
        # adjust the title position and size
        column_title_gp = gpar(fontsize = 16, fontface = "bold", hjust = 0.5),

    )

png(organoid_consensus_heatmap_file_path, width = width, height = height, units = "in", res = 600)
# save as a PNG
draw(heatmap_plot, merge_legend = TRUE, heatmap_legend_side = "right")
dev.off()
heatmap_plot

head(organoid_fs_df)


# drop columns that contain neighbors
organoid_fs_df <- organoid_fs_df %>%
  select(-contains("Neighbors")) %>%
  select(-contains("single_cell_count"))

dim(organoid_fs_df)
# drop all rows that have any NA values
organoid_fs_df <- organoid_fs_df %>%
  filter(complete.cases(.))
dim(organoid_fs_df)

column_anno <- HeatmapAnnotation(
    Target = organoid_fs_df$Target,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    col = list(
            Target = custom_MOA_palette
        )
)

patient_columns_anno <- HeatmapAnnotation(
    Patient = organoid_fs_df$patient,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16)))
)

depth_anno <- HeatmapAnnotation(
    Depth = organoid_fs_df$Area.Size.Shape_Organoid_CENTER.Z,
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 16),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16)))
)

column_annotations = c(column_anno, patient_columns_anno, depth_anno)


# get the list of features
features <- colnames(organoid_fs_df)
features <- features[!features %in% c("treatment", "Target", "Class", "single_cell_count", "patient",
         "object_id",
         "unit",
         "dose",
         "image_set",
         "Well",
         "single_cell_count",
         "Area.Size.Shape_Organoid_CENTER.X",
         "Area.Size.Shape_Organoid_CENTER.Y",
         "Area.Size.Shape_Organoid_CENTER.Z"
         )]
features <- as.data.frame(features)
rownames(features) <- features$features
# split the features by _ into multiple columns
features <- features %>%
    separate(features, into = c("Feature Type", "Compartment", "Channel", "Measurement"), sep = "_", extra = "merge", fill = "right")
# if Feature type is AreaSizeShape then shift the Channel to the Measurement column and set Channel to NA
features <- features %>%
    mutate(
        Measurement = ifelse(`Feature Type` == "Area.Size.Shape", Channel, Measurement)
    )
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Area.Size.Shape", "None", Channel)
    )

# select the first channel for colocalization features channels are split by .
features <- features %>%
    mutate(
        Channel = ifelse(`Feature Type` == "Colocalization",
                         sub("\\..*", "", Channel),
                         Channel)
    )


# sort by feature type
features <- features %>%
    arrange(`Feature Type`, Compartment, Channel, Measurement)

row_measurement = rowAnnotation(
    FeatureType = features$`Feature Type`,
           annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))),
    annotation_name_side = "bottom",
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
            FeatureType = c(
            "Area.Size.Shape" = brewer.pal(8, "Paired")[1],
            "Colocalization" = brewer.pal(8, "Paired")[2],
            "Granularity" = brewer.pal(8, "Paired")[3],
            "Intensity" = brewer.pal(8, "Paired")[4],
            "Texture" = brewer.pal(8, "Paired")[8]
        )
    ),
    show_legend = TRUE
)
row_channel = rowAnnotation(
    Channel = features$Channel,
        annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        # make annotation bar text bigger
        legend = gpar(fontsize = 16),
        annotation_name = gpar(fontsize = 16),
        # legend_height = unit(20, "cm"),
        legend_width = unit(1, "cm"),
        # make legend taller
        # legend_height = unit(10, "cm"),
        legend_width = unit(1, "cm"),
        legend_key = gpar(fontsize = 16)
        )
    ),



    annotation_name_side = "bottom",
    # make font size bigger
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
    Channel = c(
            "DNA" = "#0000AB",
            "AGP" = "#b1001a",
            "Mito" = "#B000B0",
            "ER" = "#00D55B",
            "BF" = "#FFFF00",
            "None" = "#B09FB0")
    )
)
row_annotations = c(row_measurement, row_channel)

mat <- organoid_fs_df %>%
  select(-treatment, -Class, -Target, -patient, -object_id, -image_set, -Well,
         -unit,
         -dose,
         -Area.Size.Shape_Organoid_CENTER.X,
         -Area.Size.Shape_Organoid_CENTER.Y,
         -Area.Size.Shape_Organoid_CENTER.Z
         ) %>%

  as.matrix()
mat <- t(mat)
colnames(mat) <- organoid_fs_df$treatment
dim(mat)

width <- 20
height <- 15
options(repr.plot.width = width, repr.plot.height = height)
heatmap_plot <- Heatmap(
        mat,
        # col = col_fun,
        show_row_names = FALSE,
        # cluster_columns = FALSE,
        show_column_names = FALSE,
        row_dend_reorder = TRUE, # reorder rows based on dendrogram
        column_dend_reorder = TRUE, # reorder columns based on dendrogram
        show_row_dend = FALSE,
        show_column_dend = FALSE,

        column_names_gp = gpar(fontsize = 16), # Column name label formatting
        row_names_gp = gpar(fontsize = 14),

        # show_heatmap_legend = FALSE,
        heatmap_legend_param = list(
                    title = "Norm\nValue",
                    title_position = "topcenter",
                    title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
                    labels_gp = gpar(fontsize = 16),
                    # legend_height = unit(2, "cm"),
                    legend_width = unit(3, "cm"),
                    annotation_legend_side = "bottom"
                    ),

        row_dend_width = unit(4, "cm"),
        column_dend_height = unit(4, "cm"),
        # column_title = paste0("Dose: ", dose," uM"),
        right_annotation = row_annotations,
        top_annotation = column_annotations,
        # adjust the title position and size
        column_title_gp = gpar(fontsize = 16, fontface = "bold", hjust = 0.5),

    )
png(organoid_fs_heatmap_file_path, width = width, height = height, units = "in", res = 600)
# save as a PNG
draw(heatmap_plot, merge_legend = TRUE, heatmap_legend_side = "right")
dev.off()
heatmap_plot
