# Shiny App Deployment Instructions

You will need to create a `.Renviron` file with the necessary environment variables.
The file should be located: `6.dynamic_viz_Rshiny/.Renviron`
I have provided a template file named `.Renviron_template` in the same directory.
You can copy this file to create your own `.Renviron` file:
This file should contain three lines with your RStudio Connect account details:
```bash
source 6.dynamic_viz_Rshiny/deploy_app.sh
```

You will also need to set up the `.Renviron` file with the necessary environment variables. This file should contain:
```
RSCONNECT_NAME="your_account_name
RSCONNECT_TOKEN="your_token"
RSCONNECT_SECRET="your_secret"
```
⚠️ **Critical Note:**
Make sure to replace `your_account_name`, `your_token`, and `your_secret` with your actual RStudio Connect account details.
This file is used to authenticate your deployment to the shiny server. Make sure to replace `your_account_name`, `your_token`, and `your_secret` with your actual RStudio Connect account details.
This file should not be committed to version control as it contains sensitive information.
Make sure to double check that the `.Renviron` file is added to your `.gitignore` file to prevent it from being tracked by git.


To deploy the shiny app, run the following command:
```bash
source 6.dynamic_viz_Rshiny/deploy_app.sh
```


Note: The app is optionally hosted for free on shinyapps.io, which is a service provided by Posit.
The free tier has some limitations, such as a maximum of 25 active hours per month and a maximum of 5 applications.
You can upgrade to a paid plan if you need more resources or features.
