library(shiny)

# Source UI and Server components
source("ui.R")
source("server.R")

# Create and run the Shiny app
shinyApp(ui = ui, server = server)
