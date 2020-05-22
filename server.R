library(shiny)
library(shinydashboard)
library(plotly)
library(shinyWidgets)
library(shinyjs)
library(shinyBS)
library(shinyFiles)
library(gridExtra)
library(png)
library(grid)
library(shinyalert)
library(shinycustomloader)
library(reticulate)

server <- function(input, output, session) {

  s_user <- tags$div(id = "s_user",
                      shinyjs::useShinyjs(),
                      fluidRow(
                        column(width = 4, br(),
                               fileInput(inputId = "t_files",
                                         label = "Select 4-5 Genuine Signatures",
                                         multiple = TRUE, accept=c('.png'), width = '100%', 
                                         buttonLabel = "Browse...",
                                         placeholder = "Choose PNG images")),
                        column(width = 3, br(),
                               textInput(inputId = "cust", label = "CustomerID",
                                         width = "90%", placeholder = "No Space allowed")),
                        column(width = 2, br(), br(), 
                               actionButton("process1", "Train Data", icon = icon("spinner"), width = "85%")),
                        column(width = 3, br(),
                               helpText(HTML("<em>Note</em>: After Uploading the files, click "),
                                        tags$b("Train Data"),
                                        "to process the signatures", br(), br(), 
                                        "Atleast 4 signatures are required per Customer"))
                      )
  )
  
  m_user <- tags$div(id = "m_user",
                     shinyjs::useShinyjs(),
                     fluidRow(
                       column(width = 6,  br(),
                              fileInput(inputId = "dir",
                                        label = "Choose Input Signatures",
                                        multiple = TRUE, accept=c('.png'), width = '90%', 
                                        buttonLabel = "Browse...",
                                        placeholder = "Choose PNG images")
                              ),
                       column(width = 2, br(), br(), 
                              actionButton("process2", "Train Dataset",
                                                                   icon = icon("spinner"), width = "85%")),
                       column(width = 4,
                              helpText(HTML("<em>Note</em>: Please ensure that the signatures selected are in PNG format and naming pattern of signatures is custid_partno.png for eg. <em>'Customer01_1.png'</em>"), br(),
                                       "After selecting all the input signatures, click ",
                                       tags$b("Train Dataset"),
                                       "to process the signatures", br(), br(), 
                                       "Atleast 4 signatures are required per Customer"))
                     )
  )

  user_del <- tags$div(id = "user_del",
                     shinyjs::useShinyjs(),
                     fluidRow(
                       column(width = 8,
                              pickerInput(
                                inputId = "to_rm", label = "Select the Customers to be deleted:",
                                choices = list.dirs(path = data_path, full.names = F, recursive = F),
                                selected = NULL, width = "100%", multiple = T,
                                options = list(`actions-box` = TRUE, liveSearch = T, 
                                               selectedTextFormat = 'count > 9', virtualScroll = 10,
                                               width = 'auto')
                              )
                        ),
                       column(width = 2, offset = 2, br(),
                              actionButton("delete", tags$b("Delete Customers"),
                                            icon = icon("times"), width = "90%"))
                     )
  )
  
  output$d_update <- renderUI({
    if(input$n_user == "Single User Addition")
      s_user
    else if(input$n_user == "Multiple Users Addition")
      m_user
    else
      user_del
  })
  
  observeEvent(input$process1, {
    
    inFile <- input$t_files
    if(is.null(inFile))
      return(NULL)
    else
    {
      files <- inFile$datapath
      if(length(files) < 4)
        shinyalert(text = "Please Select atleast 4 signature images to train the model", type = "error",
                   closeOnClickOutside = T, showCancelButton = F, showConfirmButton = F)
      else
      {
        cust <- input$cust
        output$text <- renderUI({
          custID <- single_user_add(img_paths = files, out_path = data_path, cust_id = cust)
          updatePickerInput(session, inputId = "to_rm",
                            choices = list.dirs(path = data_path, full.names = F, recursive = F))
          updateSelectInput(session, "cust_test", choices = c("",list.dirs(path = data_path, full.names = F, recursive = F)))
          h5(paste0("CustomerID ", custID," is successfully added to database"))
        })
        showModal(modalDialog(withLoader(uiOutput('text'))))
      }
    }
  })
  
  observeEvent(input$process2, {
    
    inPath <- input$dir
    if(is.null(inPath))
      return(NULL)
    else
    {
      files <- inPath$datapath
      output$text <- renderText({
        vals <- multi_user_add(src_path = files, out_path = data_path)
        updatePickerInput(session, inputId = "to_rm",
                          choices = list.dirs(path = data_path, full.names = F, recursive = F))
        updateSelectInput(session, "cust_test", choices = c("",list.dirs(path = data_path, full.names = F, recursive = F)))
        HTML(h5(paste0(vals[1]," new records added to database successfully,")),
             h5(paste0(vals[2]," existing records overwritten, &")),
             h5(paste0(vals[3]," records could not be processed due to less than 4 signatures")))
        showModal(modalDialog(withLoader(uiOutput('text'))))
      })
    }
  })

  observeEvent(input$delete, {
    ids <- input$to_rm
    showModal(modalDialog(
      h5(paste0("Are you sure, you want to delete ",length(ids),ifelse(length(ids)>1," records"," record"))), 
      title="Delete Records",
      footer = tagList(actionButton("confirmDelete", "Delete"),
                       modalButton("Cancel"))
    ))
  })
  
  observeEvent(input$confirmDelete,{
    ids <- input$to_rm
    rm_users(data_path = data_path, cust_ids = ids)
    removeModal()
    showModal(modalDialog(
      title = NULL,
      h5(paste0(length(ids)," Customer ",ifelse(length(ids)>1,"records","record"), " successfully removed from the database")),
      easyClose = TRUE,
      footer = NULL
    ))
    updatePickerInput(session, inputId = "to_rm",
                      choices = list.dirs(path = data_path, full.names = F, recursive = F))
    updateSelectInput(session, "cust_test", choices = c("",list.dirs(path = data_path, full.names = F, recursive = F)))
  })
  
  
  output$train <- renderUI({
    box(title = ifelse(input$cust_test=="","Genuine Signatures",paste(input$cust_test," Genuine Signatures")), 
        background = NULL, width = "100%", height = "644px", solidHeader = T, status = "info",
        uiOutput("cust_orig_sign",inline = F)
        )
  })
  
  output$cust_orig_sign <- renderUI({
      if(input$cust_test != "")
        tags$div(
          column(width = 9, offset = 3,
                 radioButtons(inputId = "orig_sig_type", choices = c("Original","Processed"), 
                              selected = "Original", inline = T, width = "100%", label = NULL)),
          box(width=12, height = NULL, align = "center",
              renderPlot({
                filename <- normalizePath(list.files(path = file.path(data_path,input$cust_test,tolower(input$orig_sig_type)), full.names = T, recursive = F, pattern = ".png"))
                pngs = lapply(filename, readPNG)
                asGrobs = lapply(pngs, rasterGrob, x = unit(0.5, "npc"), y = unit(0.5, "npc"))
                p <- grid.arrange(grobs=asGrobs, ncol = 1)
              }, width = 500, height = 500))
        )
    else
      return(NULL)
  })
    
  output$test <- renderUI({
    tags$div(
      box(title = "Uploaded Signature", solidHeader = T, width = "100%", height = "200px", status = "info",
          plotOutput("test_orig"), align = "center"),
      box(title = "Processed Signature", solidHeader = T, width = "100%", height = "200px", status = "info",
          imageOutput("test_proc"), align = "center"),
      box(title = "Match Score", solidHeader = T, width = "100%", height = "200px", status = "info",
          uiOutput("score"))
    )
  })
  
  scoring <- reactive({
    inFile <- input$test_path
    if(is.null(inFile))
      return(NULL)
    else
    {
      if(input$criteria_test == "Strict")
        pval <- 0.25
      else if(input$criteria_test == "Normal")
        pval <- 0.5
      else
        pval <- 0.75
      
      return(testing(cust = input$cust_test,train_path = data_path,img_path = inFile$datapath,method="Manhattan",p=pval))
    }
  })
    
    
  observeEvent(input$run_test, {
    
    inFile <- input$test_path
    if(is.null(inFile))
      return(NULL)
    else
    {
      output$test_orig <- renderPlot({
        filename <- inFile$datapath
        png = readPNG(filename)
        asGrobs = grid.raster(png, x = unit(0.5, "npc"), y = unit(0.5, "npc"))
      }, width = 500, height = 130)

      score  <- scoring()
      
      output$test_proc <- renderPlot({
        filename <- file.path(data_path,input$cust_test,"test.png")
        png = readPNG(filename)
        asGrobs = grid.raster(png, x = unit(0.5, "npc"), y = unit(0.5, "npc"))
      }, width = 500, height = 130)

      output$score <- renderUI({
                          valueBox(paste0(round(score,2),"%"), subtitle = "Percentage match with the Genuine Signatures", width = 12,
                                   color = ifelse(score<50,"red","green"))
                      })
    }
  })
  
  observeEvent(input$reset, {
    updateSelectInput(session, "cust_test", selected = "")
    updateSelectInput(session, "criteria_test", selected = "Normal")
    output$score <- NULL
    output$test_proc <- NULL
    output$test_orig <- NULL
    reset("test_path")
  })

  observeEvent(input$cust_test, {
    updateSelectInput(session, "criteria_test", selected = "Normal")
    output$score <- NULL
    output$test_proc <- NULL
    output$test_orig <- NULL
    reset("test_path")
  })
} 