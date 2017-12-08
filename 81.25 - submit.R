rm(list =  ls())

data1 = read.csv("train_u6lujuX_CVtuZ9i.csv",na.strings = c("","NA"))
eval = read.csv("test_Y3wMUE5_7gLdaTN.csv",na.strings = c("","NA"))
evalid = eval[,1]
str(data1)
str(eval)
summary(data1)
summary(eval)
sum(is.na(data1))

sapply(data1, function(x) sum(is.na(x)))
sapply(eval, function(x) sum(is.na(x)))

library(DMwR)
library(infotheo)

table(data1$Gender)
table(data1$Credit_History)
table(data1$Loan_Amount_Term)
table(eval$Credit_History)

data1$Credit_History = as.factor(as.character(data1$Credit_History))
eval$Credit_History = as.factor(as.character(eval$Credit_History))

data1 = knnImputation(data1, k = 5)
eval = knnImputation(eval, k = 5)

data1$Loan_Status = ifelse(data1$Loan_Status=="N",0,1)

data1$Loan_Status = as.factor(as.character(data1$Loan_Status))

str(data1)

data1 = data1[,-1]
eval = eval[,-1]

#############################
#Feautre Engineering
#############################

data1$Loan_Amount_Term = data1$Loan_Amount_Term / 12
eval$Loan_Amount_Term = eval$Loan_Amount_Term / 12

data1$Total_Income = data1$ApplicantIncome + data1$CoapplicantIncome
eval$Total_Income = eval$ApplicantIncome + eval$CoapplicantIncome

data1$LoanAmount_Month = data1$LoanAmount / data1$Loan_Amount_Term
eval$LoanAmount_Month = eval$LoanAmount / eval$Loan_Amount_Term

library(corrgram)

corrgram(data1)
cor(data1$ApplicantIncome,data1$LoanAmount)

#############################
#Random Forest
#############################
library(randomForest)
library(caret)
set.seed(114)
RF = randomForest(Loan_Status~.,data = data1,keep.forest = TRUE,ntree = 50)
summary(RF)
RF$importance
varImpPlot(RF)
Random_Forest = RF$predicted #prediction of train set
CM_rf = table(RF$predicted,data1$Loan_Status)
CM_rf
confusionMatrix(CM_rf)

CM_rf_eval = predict(RF,eval)

Loan_Status = data.frame(CM_rf_eval)
Random_Forest_eval = Loan_Status #prediction of eval set

Loan_Status = ifelse(CM_rf_eval==1,"Y","N")
Loan_Status = data.frame(Loan_Status)

Loan_Status = cbind(evalid,Loan_Status)
write.csv(Loan_Status,file = "submission.csv",row.names = FALSE)

library(vegan)

#############################
#Logistic Regression
#############################
set.seed(114)
data_glm = data1

eval_glm = eval
data_glm[,c(6,7,8,9,13,14)] = decostand(data_glm[,c(6,7,8,9,13,14)],"range")
eval_glm[,c(6,7,8,9,12,13)] = decostand(eval_glm[,c(6,7,8,9,12,13)],"range")

str(data_glm)
summary(data_glm)

Loan_glm <- glm(data_glm$Loan_Status~., data = data_glm, family = binomial)
summary(Loan_glm)

library(MASS)

stepAIC(Loan_glm)

Loan_glm_2 = glm(data_glm$Loan_Status ~ Married + Education + LoanAmount + 
                   Credit_History + Property_Area, family = binomial, data = data_glm)
summary(Loan_glm_2)

predict_glm <- predict(Loan_glm_2, newdata = data_glm, type = 'response')
pred_class <- ifelse(predict_glm >0.5,1,0)
GLM = pred_class #prediction on train set
confusionMatrix(pred_class,data_glm$Loan_Status)


library(ggplot2)
library(ROCR)
ROCRpred = prediction(predict_glm, data_glm$Loan_Status)
as.numeric(performance(ROCRpred, "auc")@y.values)
ROCRperf <- performance(ROCRpred, "tpr", "fpr")
par(mfrow=c(1,1))
plot(ROCRperf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

#Tweak
pred_class1 <- ifelse(predict_glm >0.6,1,0)
confusionMatrix(pred_class1,data_glm$Loan_Status)

#Test
predict_test_glm <- predict(Loan_glm_2, newdata = eval_glm, type = 'response')
Loan_Status_glm <- ifelse(predict_test_glm>0.5, 1, 0)
GLM_eval = Loan_Status_glm
table(Loan_Status_glm)
GLM_eval = ifelse(GLM_eval==1,"Y","N")
Loan_Log<-data.frame(GLM_eval)
submission_glm <- cbind(evalid,Loan_Log)

write.csv(submission_glm,file = "submission_glm.csv",row.names = FALSE)

#############################
#ADABOOST
#############################

library(ada)

str(data1)

ada_data = data1
ada_eval = eval


ada_data[,c(6,7,8,9,13,14)] = decostand(ada_data[,c(6,7,8,9,13,14)],"range")
ada_eval[,c(6,7,8,9,12,13)] = decostand(ada_eval[,c(6,7,8,9,12,13)],"range")

str(ada_eval)
str(ada_data)

ada_data_WT = ada_data[,-12]
ada_data_T = ada_data[,12]
set.seed(114)
model_ada = ada(ada_data_WT,ada_data_T,iter = 90)

pred_ada = predict(model_ada,ada_data_WT) #Prediction on train
result_ada = table(pred_ada,ada_data_T)
confusionMatrix(result_ada)

pred_eval_ada = predict(model_ada,ada_eval)

pred_eval_ada = ifelse(pred_eval_ada==1,"Y","N")
pred_eval_ada = data.frame(pred_eval_ada)

submission_ada <- cbind(evalid,pred_eval_ada)

write.csv(submission_ada,file = "submission_ada.csv",row.names = FALSE)


### Stacking
#Mode stack (RF,GLM,ADA)

Mode <- function(x) {
  ux<- unique(x)
  ux[which.max(tabulate(match(x, ux)))] }

#eval_mode_stack = eval_stack (Including NN trying for Log instead that's y commenting it out)
eval_mode_stack = cbind(GLM_eval,Random_Forest_eval,pred_eval_ada)

str(eval_mode_stack)

eval_mode_stack$CM_rf_eval = ifelse(eval_mode_stack$CM_rf_eval ==0,0,1)
eval_mode_stack$GLM_eval = ifelse(eval_mode_stack$GLM_eval =="Y",1,0)
eval_mode_stack$pred_eval_ada = ifelse(eval_mode_stack$pred_eval_ada =="Y",1,0)

eval_mode_stack$Final = apply(eval_mode_stack, 1, Mode)

Loan_Mode = eval_mode_stack$Final

Loan_Mode = ifelse(Loan_Mode==1,"Y","N")


Loan_Mode = data.frame(Loan_Mode)

Loan_Status_Stack_Mode = cbind(evalid,Loan_Mode)
setnames(Loan_Status_Stack_Mode,c("evalid","Loan_Mode"),c("Loan_ID","Loan_Status"))
write.csv(Loan_Status_Stack_Mode,file = "submission_mode_stack.csv",row.names = F)

