library(shiny)
library(shinydashboard)
library(plotly)
library(shinyWidgets)
library(shinyjs)
library(shinyalert)
library(shinydashboardPlus)

ui <- dashboardPage(
  dashboardHeader(title = "Signature Verification",titleWidth = "100%"),
  dashboardSidebar(disable = T),
  dashboardBody(
    shinyjs::useShinyjs(),
    useShinyalert(),
    fluidPage(
      box(width = 12, title = "Dataset Updation", status = "primary",
          solidHeader= TRUE, collapsible = T, collapsed = T,
          column(width = 12,
                 tags$style(HTML(".radio-inline {margin-right: 42px;}")),
                 radioButtons(inputId = "n_user", label = "Select appropriate action you want to perform", choices = c("Single User Addition","Multiple Users Addition","Users Deletion"), selected = "Single User Addition", inline = T, width = "100%")),
          column(width = 12, uiOutput("d_update"))
      ),
      box(width = 12, title = "Signature Testing", status = "primary",
          solidHeader= TRUE, collapsible = F,
          column(width = 12,
                 box(title = tags$b("Testing Parameters"), status = NULL, solidHeader = F,
                     background = "aqua", width = NULL, height = NULL, collapsible = F,
                     column(width = 3,
                            selectInput("cust_test", "Select Customer", 
                                        choices = c("",list.dirs(path = data_path, full.names = F, recursive = F)),
                                        multiple = F, width = "100%")
                     ),
                     column(width = 2,
                            selectInput("criteria_test", "Select Testing Criteria", 
                                        choices = c("Relaxed","Normal","Strict"), selected = "Normal",
                                        multiple = F, width = "100%")
                     ),
                     column(width = 5,
                            fileInput(inputId = "test_path", label = "Test Signature", multiple = F,
                                      accept = c(".png"), width = "100%", buttonLabel = "Browse...",
                                      placeholder = "No File Selected")
                     ),
                     column(width = 2,
                            column(width=12, actionButton(inputId = 'run_test', label = tags$b("Check"),
                                         icon = icon("cogs"), width = "85%")),
                            column(width=12, br()),
                            column(width=12, actionButton(inputId = 'reset', label = tags$b("Reset"),
                                         icon = icon("refresh"), width = "85%"))
                     )
                     
                  )
          ),
          column(width = 6, uiOutput("train")),
          column(width = 6, uiOutput("test"))
      )
    )
  )
)

