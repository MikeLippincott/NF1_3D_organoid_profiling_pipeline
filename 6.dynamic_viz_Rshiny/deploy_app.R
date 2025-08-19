library(dplyr)
library(ggplot2)
library(shiny)
library(rsconnect)
library(argparser)

# Parse command line arguments
parser <- argparser::arg_parser(description = "Deploy R Shiny app")
parser <- argparser::add_argument(parser, "--testing", flag = TRUE, help = "Run in testing mode without deployment")
args <- argparser::parse_args(parser)
testing <- args$testing

if (!testing) {

    # Load environment variables from .Renviron file
    # This happens automatically when R starts, but we can explicitly load it
    readRenviron(".Renviron")

    # Optional: Check if environment variables are loaded correctly
    cat("RSCONNECT_NAME:", Sys.getenv("RSCONNECT_NAME"), "\n")
    cat("Token loaded:", ifelse(nchar(Sys.getenv("RSCONNECT_TOKEN")) > 0, "Yes", "No"), "\n")
    cat("Secret loaded:", ifelse(nchar(Sys.getenv("RSCONNECT_SECRET")) > 0, "Yes", "No"), "\n")


    rsconnect::setAccountInfo(
    name = Sys.getenv("RSCONNECT_NAME"),
    token = Sys.getenv("RSCONNECT_TOKEN"),
    secret = Sys.getenv("RSCONNECT_SECRET")
    )

    rsconnect::deployApp(
        appDir = "./",
        appName = "NF1_Patient_Organoid_Dashboard",
        account = Sys.getenv("RSCONNECT_NAME"),
        launch.browser = TRUE
        )
} else {

    # Or completely disable browser launch
    runApp("app.R", launch.browser = FALSE)
}
