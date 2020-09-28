library(readr)
library(ggplot2)
library(GGally)
library(caret) # models
library(corrplot) # correlation plots
library(DALEX) # explain models
library(DescTools) # plots
library(doParallel) # parellel processing
library(dplyr) # syntax
library(GGEBiplots) # PCA plots
install.packages("ggbiplot")
install.packages("GGEBiplots")
library(inspectdf) # data overview
library(readr) # quick load
library(sjPlot) # contingency tables
library(tabplot) # data overview
library(tictoc) # measure time

data<-read_csv("C:/Users/sneha/Downloads/bank (1).csv")
summary(data)  
is.null(data)
is.na(data)
summary(data)
head(data)
prop.table(table(data$poutcome))
clean1<-data[-16]
prop.table(table(clean1$deposit))
prop.table(table(clean1$))
clean1$default<-as.numeric(supressWarning(clean1$default))
typeof(default)

#attach(clean1)
typeof(deposit)


for (i in 1:nrow(clean1)) {
  if (clean1[i,16] == 'yes') {
    clean1[i,16] <- 1
  }
  else {
    clean1[i,16] <- 0
  }
}

df_cat <- select_if(data, is.character) %>% names()
# remove the response
response_ind <- match('deposit', df_cat)
df_cat <- df_cat[-response_ind]

# plot categorical variables
for (i in df_cat) {
  print(i)
  
  print(
    sjp.xtab(df$deposit,
             df[[i]],
             margin = "row",
             bar.pos = "stack",
             axis.titles = "deposit",
             legend.title = i)
  )
}


boxplot(data$poutcome)
pro


data[17] <- as.numeric(as.character(data$default))
