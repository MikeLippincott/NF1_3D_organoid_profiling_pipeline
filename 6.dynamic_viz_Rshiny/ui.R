# Load necessary libraries
library(shiny)


# Define UI
ui <- fluidPage(
  titlePanel("UMAP Plot"),

  sidebarLayout(
    sidebarPanel(
      uiOutput("sc_or_organoid"),
      uiOutput("data_level"),
      uiOutput("PatientSelect"),
      uiOutput("TreatmentSelect"),
      uiOutput("ColorBySelect"),
      uiOutput("FacetSelect")
    ),

    mainPanel(
      plotOutput("umapPlot", width = "100%", height = "900px")
    )
  )
)
